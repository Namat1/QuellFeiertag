from __future__ import annotations

import io
import json
import math
from datetime import datetime
from html import escape
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pgeocode
except ImportError:
    pgeocode = None

st.set_page_config(page_title="Feiertags-Tourenplaner – HTML Generator", page_icon="🧭", layout="wide")

SHEETS = ["DIREKT", "MK", "HUPA_NMS", "HUPA_MALCHOW"]
DAY_COLUMNS = ["Mo", "Die", "Mitt", "Don", "Fr", "Sam"]
DAY_LABELS = {"Mo": "Montag", "Die": "Dienstag", "Mitt": "Mittwoch", "Don": "Donnerstag", "Fr": "Freitag", "Sam": "Samstag"}
DEFAULT_DEPOT_PLZ = {"DIREKT": "24539", "MK": "24539", "HUPA_NMS": "24539", "HUPA_MALCHOW": "17213"}

st.markdown(
    """
<style>
.block-container{max-width:1200px;padding-top:1.4rem}
.small{color:#9ca3af;font-size:.9rem}
div[data-testid="stMetricValue"]{font-size:1.5rem}
</style>
""",
    unsafe_allow_html=True,
)


def clean_plz(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")
    return s.str.zfill(5)


@st.cache_data(show_spinner=False)
def read_source(file_bytes: bytes) -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    missing = [s for s in SHEETS if s not in xl.sheet_names]
    if missing:
        raise ValueError("Fehlende Blätter: " + ", ".join(missing))

    frames: List[pd.DataFrame] = []
    for sheet in SHEETS:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
        df = df.rename(columns={"Straße": "Strasse"})
        cols = ["CSB", "SAP", "Name", "Strasse", "Plz", "Ort", *DAY_COLUMNS]
        for col in cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[cols].copy()
        df["Quelle"] = sheet
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["SAP"] = pd.to_numeric(out["SAP"], errors="coerce").astype("Int64")
    out["CSB"] = pd.to_numeric(out["CSB"], errors="coerce").astype("Int64")
    out["Plz"] = clean_plz(out["Plz"])
    for d in DAY_COLUMNS:
        out[d] = pd.to_numeric(out[d], errors="coerce").astype("Int64")

    out = out.dropna(subset=["SAP", "Name", "Plz"]).copy()
    out["uid"] = out.apply(
        lambda r: f"{r['Quelle']}::{int(r['SAP'])}::{int(r['CSB']) if pd.notna(r['CSB']) else 'x'}",
        axis=1,
    )
    return out


@st.cache_resource(show_spinner=False)
def get_geocoder():
    if pgeocode is None:
        return None
    return pgeocode.Nominatim("de")


@st.cache_data(show_spinner=False)
def geocode_plz(postcodes: tuple[str, ...]) -> pd.DataFrame:
    geo = get_geocoder()
    if geo is None:
        return pd.DataFrame(columns=["Plz", "lat", "lon"])
    vals = sorted({str(x).zfill(5) for x in postcodes if x and x != "<NA>"})
    if not vals:
        return pd.DataFrame(columns=["Plz", "lat", "lon"])
    r = geo.query_postal_code(vals)
    if isinstance(r, pd.Series):
        r = r.to_frame().T
    return pd.DataFrame({
        "Plz": pd.Series(vals, dtype="string"),
        "lat": pd.to_numeric(r["latitude"], errors="coerce").to_numpy(),
        "lon": pd.to_numeric(r["longitude"], errors="coerce").to_numpy(),
    })


def enrich_geo(df: pd.DataFrame) -> pd.DataFrame:
    coords = geocode_plz(tuple(df["Plz"].dropna().astype(str).unique().tolist()))
    return df.merge(coords, on="Plz", how="left")


def coord_for_plz(plz: str) -> dict:
    x = geocode_plz((str(plz).zfill(5),))
    if x.empty or x[["lat", "lon"]].isna().any(axis=None):
        return {"lat": None, "lon": None}
    return {"lat": float(x.iloc[0]["lat"]), "lon": float(x.iloc[0]["lon"])}


def json_ready(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def make_payload(df: pd.DataFrame) -> list[dict]:
    keep = ["uid", "Quelle", "CSB", "SAP", "Name", "Strasse", "Plz", "Ort", *DAY_COLUMNS, "lat", "lon"]
    rows = []
    for rec in df[keep].to_dict("records"):
        rows.append({k: json_ready(v) for k, v in rec.items()})
    return rows


def build_html(customers: list[dict], depot_cfg: dict, source_name: str) -> str:
    data_json = json.dumps(customers, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    dep_json = json.dumps(depot_cfg, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    src = escape(source_name)
    generated = datetime.now().strftime("%d.%m.%Y %H:%M")

    return f'''<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feiertags-Tourenplaner</title>
<style>
:root{{--bg:#101114;--panel:#181a20;--panel2:#20232b;--line:#30343d;--text:#f2f3f5;--muted:#9ca3af;--accent:#9d6cff;--accent2:#ff9a3c;--good:#66bb6a;--bad:#ef6464}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(16,17,20,.95);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}
.wrap{{max-width:1800px;margin:auto;padding:18px 22px}} h1{{font-size:25px;margin:0 0 3px}} .sub{{color:var(--muted)}}
.controls{{display:grid;grid-template-columns:1.3fr 1.2fr .8fr .9fr auto auto;gap:12px;align-items:end;margin-top:15px}} .box{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}}
label.title{{display:block;color:var(--muted);font-size:12px;margin-bottom:7px}} .checks{{display:flex;gap:8px;flex-wrap:wrap}} .chip{{display:inline-flex;gap:5px;align-items:center;background:#242730;border:1px solid #343844;border-radius:999px;padding:5px 8px}}
input[type=number],input[type=date]{{width:100%;background:#111318;border:1px solid #3a3e48;color:var(--text);border-radius:8px;padding:8px}}
button{{background:#2a2d36;border:1px solid #404550;color:var(--text);border-radius:9px;padding:9px 12px;cursor:pointer;font-weight:650}} button:hover{{filter:brightness(1.12)}} button.primary{{background:var(--accent);border-color:var(--accent)}} button.warn{{background:#3a2528;border-color:#5a3035;color:#ffb6bd}}
main{{max-width:1800px;margin:auto;padding:18px 22px 50px}} .metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}} .metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 12px}} .metric b{{display:block;font-size:20px}} .metric span{{color:var(--muted);font-size:12px}}
.tabs{{display:flex;gap:7px;margin:8px 0 14px}} .tabbtn.active{{background:var(--accent)}} .tab{{display:none}} .tab.active{{display:block}}
.board{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px;align-items:start}} .route{{background:var(--panel);border:1px solid var(--line);border-radius:13px;overflow:hidden;min-height:150px}} .routeHead{{padding:10px 11px;background:var(--panel2);border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px;align-items:center}} .routeTitle{{font-weight:800}} .stats{{color:var(--muted);font-size:12px;margin-top:2px}} .routeBtns{{display:flex;gap:5px}} .routeBtns button{{padding:5px 7px;font-size:12px}}
.dropzone{{min-height:90px;padding:8px}} .cust{{background:#262933;border:1px solid #383d48;border-radius:9px;padding:8px 9px;margin:6px 0;cursor:grab;display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:start}} .cust.dragging{{opacity:.35}} .handle{{color:#7f8693;font-weight:900;letter-spacing:-1px}} .cust strong{{font-size:13px}} .meta{{color:var(--muted);font-size:11px;margin-top:2px}} .del{{padding:2px 6px;border-radius:7px;background:transparent;border-color:#4a3b40;color:#d99}}
.route.unplanned{{border-color:#5a3a40}} .route.unplanned .routeHead{{background:#2b1e22}}
.toolbar{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}} .note{{color:var(--muted);font-size:12px;margin:8px 0}}
svg{{width:100%;height:640px;background:#14161b;border:1px solid var(--line);border-radius:14px}} .legend{{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}} .legend i{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}} th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;font-size:12px}} th{{position:sticky;top:0;background:#242730}} .tablewrap{{max-height:700px;overflow:auto;border:1px solid var(--line);border-radius:12px}}
@media(max-width:1050px){{.controls{{grid-template-columns:1fr 1fr}} .metrics{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:650px){{.wrap,main{{padding-left:10px;padding-right:10px}} .controls{{grid-template-columns:1fr}} .metrics{{grid-template-columns:1fr 1fr}}}}
@media print{{header,.tabs,.toolbar,.routeBtns,.del{{display:none!important}} body{{background:#fff;color:#000}} .route,.metric,table{{border:1px solid #bbb;background:#fff}} .routeHead,.cust{{background:#fff;color:#000}} .cust{{break-inside:avoid}}}}
</style>
</head>
<body>
<header><div class="wrap"><h1>🧭 Feiertags-Tourenplaner</h1><div class="sub">Quelle: {src} · erzeugt {generated} · Berechnung läuft vollständig in dieser HTML-Datei</div>
<div class="controls">
  <div class="box"><label class="title">Liefertage</label><div id="days" class="checks"></div></div>
  <div class="box"><label class="title">Bereiche</label><div id="sources" class="checks"></div></div>
  <div class="box"><label class="title">Max. Stopps je Tour</label><input id="maxStops" type="number" min="2" max="30" value="8"></div>
  <div class="box"><label class="title">Optionen</label><div class="checks"><label class="chip"><input id="sepSources" type="checkbox" checked> getrennt</label><label class="chip"><input id="returnDepot" type="checkbox" checked> Rückfahrt</label></div></div>
  <button class="primary" onclick="buildPlan()">⚡ Touren berechnen</button>
  <button onclick="addEmptyRoute()">＋ Neue Tour</button>
</div></div></header>
<main>
<div class="metrics"><div class="metric"><b id="mCustomers">0</b><span>Kunden ausgewählt</span></div><div class="metric"><b id="mRoutes">0</b><span>Touren</span></div><div class="metric"><b id="mPlanned">0</b><span>eingeplant</span></div><div class="metric"><b id="mUnplanned">0</b><span>nicht eingeplant</span></div><div class="metric"><b id="mKm">0 km</b><span>Geo-km gesamt</span></div></div>
<div class="tabs"><button class="tabbtn active" onclick="showTab('plan',this)">Touren</button><button class="tabbtn" onclick="showTab('map',this)">Geo-Ansicht</button><button class="tabbtn" onclick="showTab('table',this)">Tabelle</button></div>
<section id="tab-plan" class="tab active"><div class="toolbar"><button onclick="optimizeAll()">↻ Alle Reihenfolgen optimieren</button><button onclick="exportCsv()">⬇ CSV exportieren</button><button onclick="window.print()">🖨 Drucken</button><button class="warn" onclick="clearPlan()">Planung leeren</button></div><div class="note">Kunden mit der Maus zwischen Touren ziehen. ✕ verschiebt einen Kunden nach „Nicht eingeplant“. Geo-km sind Luftlinie; „Straßen-km ~“ ist nur eine grobe Schätzung (× 1,25).</div><div id="board" class="board"></div></section>
<section id="tab-map" class="tab"><svg id="geoSvg" viewBox="0 0 1200 640" preserveAspectRatio="xMidYMid meet"></svg><div id="legend" class="legend"></div><div class="note">Die Geo-Ansicht ist bewusst ohne Online-Kartenanbieter und funktioniert daher auch offline.</div></section>
<section id="tab-table" class="tab"><div class="tablewrap"><table><thead><tr><th>Neue Tour</th><th>Pos.</th><th>Bereich</th><th>CSB</th><th>SAP</th><th>Name</th><th>PLZ</th><th>Ort</th><th>bisherige Tour(en)</th></tr></thead><tbody id="tbody"></tbody></table></div></section>
</main>
<script>
const CUSTOMERS={data_json};
const DEPOTS={dep_json};
const DAYS={{Mo:'Montag',Die:'Dienstag',Mitt:'Mittwoch',Don:'Donnerstag',Fr:'Freitag',Sam:'Samstag'}};
const SOURCES=['DIREKT','MK','HUPA_NMS','HUPA_MALCHOW'];
const COLORS=['#9d6cff','#ff9a3c','#6fc276','#ee6363','#cf85ff','#ffc457','#84a9ff','#cd719d','#90beb2','#e6915c','#70d6ff','#ffd670'];
let selected=[]; let routes=[]; let dragUid=null;

function init(){{
  document.getElementById('days').innerHTML=Object.entries(DAYS).map(([k,v])=>`<label class="chip"><input type="checkbox" value="${{k}}" ${{k==='Fr'?'checked':''}}>${{v}}</label>`).join('');
  document.getElementById('sources').innerHTML=SOURCES.map(s=>`<label class="chip"><input type="checkbox" value="${{s}}" checked>${{s.replace('HUPA_','HUPA ')}}</label>`).join('');
  document.querySelectorAll('#days input,#sources input').forEach(x=>x.addEventListener('change',previewCount));
  previewCount();
}}
function checked(sel){{return [...document.querySelectorAll(sel+':checked')].map(x=>x.value)}}
function previewCount(){{
  const ds=checked('#days input'), ss=checked('#sources input');
  const n=CUSTOMERS.filter(c=>ss.includes(c.Quelle)&&ds.some(d=>c[d]!=null)).length;
  document.getElementById('mCustomers').textContent=n;
}}
function oldTours(c,ds){{return ds.filter(d=>c[d]!=null).map(d=>DAYS[d].slice(0,2)+' '+c[d]).join(' / ')}}
function hav(a,b){{const R=6371.0088,rad=x=>x*Math.PI/180; const dlat=rad(b.lat-a.lat),dlon=rad(b.lon-a.lon); const h=Math.sin(dlat/2)**2+Math.cos(rad(a.lat))*Math.cos(rad(b.lat))*Math.sin(dlon/2)**2; return R*2*Math.asin(Math.sqrt(h));}}
function depotFor(src){{const d=DEPOTS[src]; return d&&Number.isFinite(d.lat)&&Number.isFinite(d.lon)?d:null}}
function routeKm(items,src){{if(!items.length)return 0; const back=document.getElementById('returnDepot').checked; const dep=src==='MIX'?null:depotFor(src); let km=0; if(dep)km+=hav(dep,items[0]); for(let i=1;i<items.length;i++)km+=hav(items[i-1],items[i]); if(dep&&back)km+=hav(items[items.length-1],dep); return km}}
function proj(c,mlat){{return [c.lon*Math.cos(mlat*Math.PI/180),c.lat]}}
function clusterBalanced(arr,maxSize){{
  const n=arr.length;if(!n)return[];const k=Math.max(1,Math.ceil(n/maxSize));if(k===1)return Array(n).fill(0);
  const mlat=arr.reduce((s,c)=>s+c.lat,0)/n, pts=arr.map(c=>proj(c,mlat));
  const center=[pts.reduce((s,p)=>s+p[0],0)/n,pts.reduce((s,p)=>s+p[1],0)/n];
  const dist=(p,q)=>Math.hypot(p[0]-q[0],p[1]-q[1]);
  let seeds=[pts.map((p,i)=>[dist(p,center),i]).sort((a,b)=>b[0]-a[0])[0][1]];
  while(seeds.length<k){{let best=-1,bi=-1;for(let i=0;i<n;i++){{if(seeds.includes(i))continue;let md=Math.min(...seeds.map(s=>dist(pts[i],pts[s])));if(md>best){{best=md;bi=i}}}}seeds.push(bi)}}
  let centers=seeds.map(i=>pts[i].slice()), assign=Array(n).fill(-1); const base=Math.floor(n/k),rem=n%k,target=Array.from({{length:k}},(_,i)=>base+(i<rem?1:0));
  for(let iter=0;iter<22;iter++){{
    const D=pts.map(p=>centers.map(c=>dist(p,c))); const regret=D.map(ds=>{{let x=ds.slice().sort((a,b)=>a-b);return x[1]-x[0]}}); const order=[...Array(n).keys()].sort((a,b)=>regret[b]-regret[a]);
    const na=Array(n).fill(-1),used=Array(k).fill(0); for(const i of order){{const prefs=[...Array(k).keys()].sort((a,b)=>D[i][a]-D[i][b]);for(const c of prefs)if(used[c]<target[c]){{na[i]=c;used[c]++;break}}}}
    const same=na.every((x,i)=>x===assign[i]); assign=na; centers=centers.map((c,j)=>{{const ids=assign.map((x,i)=>x===j?i:-1).filter(i=>i>=0);return ids.length?[ids.reduce((s,i)=>s+pts[i][0],0)/ids.length,ids.reduce((s,i)=>s+pts[i][1],0)/ids.length]:c}}); if(same)break;
  }} return assign;
}}
function nearestOrder(arr,src){{
  if(arr.length<2)return arr.slice(); const rem=new Set(arr.map((_,i)=>i)),dep=src==='MIX'?null:depotFor(src); let start;
  if(dep)start=[...rem].sort((a,b)=>hav(dep,arr[a])-hav(dep,arr[b]))[0]; else{{const ctr={{lat:arr.reduce((s,c)=>s+c.lat,0)/arr.length,lon:arr.reduce((s,c)=>s+c.lon,0)/arr.length}};start=[...rem].sort((a,b)=>hav(ctr,arr[a])-hav(ctr,arr[b]))[0]}}
  const ids=[start];rem.delete(start);while(rem.size){{const cur=arr[ids[ids.length-1]],nx=[...rem].sort((a,b)=>hav(cur,arr[a])-hav(cur,arr[b]))[0];ids.push(nx);rem.delete(nx)}} let out=ids.map(i=>arr[i]);
  if(out.length>=4){{let best=routeKm(out,src),changed=true,round=0;while(changed&&round++<4){{changed=false;for(let i=0;i<out.length-2;i++)for(let j=i+2;j<out.length;j++){{const cand=out.slice(0,i).concat(out.slice(i,j).reverse(),out.slice(j));const d=routeKm(cand,src);if(d+.01<best){{out=cand;best=d;changed=true}}}}}}}}
  return out;
}}
function buildPlan(){{
  const ds=checked('#days input'),ss=checked('#sources input'),max=Math.max(2,parseInt(document.getElementById('maxStops').value||8)),sep=document.getElementById('sepSources').checked;
  if(!ds.length||!ss.length){{alert('Bitte Liefertag und Bereich auswählen.');return}}
  selected=CUSTOMERS.filter(c=>ss.includes(c.Quelle)&&ds.some(d=>c[d]!=null)).map(c=>({{...c,old:oldTours(c,ds)}}));
  const valid=selected.filter(c=>Number.isFinite(c.lat)&&Number.isFinite(c.lon)), invalid=selected.filter(c=>!Number.isFinite(c.lat)||!Number.isFinite(c.lon)); routes=[]; let num=1;
  const groups=sep?SOURCES.filter(s=>ss.includes(s)).map(s=>[s,valid.filter(c=>c.Quelle===s)]).filter(x=>x[1].length):[['MIX',valid]];
  for(const [src,grp] of groups){{const labels=clusterBalanced(grp,max);const k=Math.max(...labels,0)+1;for(let cl=0;cl<k;cl++){{let items=grp.filter((_,i)=>labels[i]===cl);items=nearestOrder(items,src);routes.push({{id:'R'+Date.now()+'_'+num,name:(sep?'FT-'+src.replace('HUPA_','')+'-':'FT-')+String(num).padStart(2,'0'),src,items}});num++}}}}
  routes.push({{id:'UNPLANNED',name:'NICHT EINGEPLANT',src:'MIX',items:invalid}}); renderAll();
}}
function renderAll(){{renderBoard();renderMetrics();renderMap();renderTable()}}
function customerHtml(c){{return `<div class="cust" draggable="true" data-uid="${{c.uid}}"><div class="handle">⋮⋮</div><div><strong>${{c.SAP}} · ${{esc(c.Ort)}} · ${{esc(c.Name)}}</strong><div class="meta">${{esc(c.Plz+' '+(c.Strasse||''))}} · ${{esc(c.old||'')}} · [${{esc(c.Quelle)}}]</div></div><button class="del" title="Nicht einplanen" onclick="event.stopPropagation();unplan('${{c.uid}}')">✕</button></div>`}}
function renderBoard(){{
  const b=document.getElementById('board'); b.innerHTML='';
  routes.forEach((r,idx)=>{{const el=document.createElement('div');el.className='route'+(r.id==='UNPLANNED'?' unplanned':'');el.dataset.route=r.id;const km=r.id==='UNPLANNED'?0:routeKm(r.items,r.src);const btns=r.id==='UNPLANNED'?'':`<div class="routeBtns"><button onclick="optimizeRoute('${{r.id}}')">↻</button><button onclick="renameRoute('${{r.id}}')">✎</button><button class="warn" onclick="removeRoute('${{r.id}}')">🗑</button></div>`;el.innerHTML=`<div class="routeHead"><div><div class="routeTitle">${{esc(r.name)}}</div><div class="stats">${{r.items.length}} Stopps${{r.id==='UNPLANNED'?'':' · '+km.toFixed(1)+' Geo-km · ~'+(km*1.25).toFixed(0)+' Straßen-km'}}</div></div>${{btns}}</div><div class="dropzone" data-route="${{r.id}}">${{r.items.map(customerHtml).join('')}}</div>`;b.appendChild(el)}}); bindDnD();
}}
function bindDnD(){{document.querySelectorAll('.cust').forEach(el=>{{el.addEventListener('dragstart',()=>{{dragUid=el.dataset.uid;el.classList.add('dragging')}});el.addEventListener('dragend',()=>el.classList.remove('dragging'));el.addEventListener('dragover',e=>e.preventDefault());el.addEventListener('drop',e=>{{e.preventDefault();moveUid(dragUid,el.closest('.dropzone').dataset.route,el.dataset.uid)}})}});document.querySelectorAll('.dropzone').forEach(z=>{{z.addEventListener('dragover',e=>e.preventDefault());z.addEventListener('drop',e=>{{if(e.target.closest('.cust'))return;e.preventDefault();moveUid(dragUid,z.dataset.route,null)}})}})}}
function findAndRemove(uid){{for(const r of routes){{const i=r.items.findIndex(c=>c.uid===uid);if(i>=0)return r.items.splice(i,1)[0]}}return null}}
function moveUid(uid,toId,beforeUid){{if(!uid)return;const c=findAndRemove(uid),to=routes.find(r=>r.id===toId);if(!c||!to)return;let ix=beforeUid?to.items.findIndex(x=>x.uid===beforeUid):-1;if(ix<0)to.items.push(c);else to.items.splice(ix,0,c);renderAll()}}
function unplan(uid){{moveUid(uid,'UNPLANNED',null)}}
function addEmptyRoute(){{if(!routes.length)routes=[{{id:'UNPLANNED',name:'NICHT EINGEPLANT',src:'MIX',items:[]}}];const un=routes.findIndex(r=>r.id==='UNPLANNED');const n=routes.filter(r=>r.id!=='UNPLANNED').length+1;routes.splice(un<0?routes.length:un,0,{{id:'R'+Date.now(),name:'FT-MAN-'+String(n).padStart(2,'0'),src:'MIX',items:[]}});renderAll()}}
function removeRoute(id){{const i=routes.findIndex(r=>r.id===id),u=routes.find(r=>r.id==='UNPLANNED');if(i<0||!u)return;u.items.push(...routes[i].items);routes.splice(i,1);renderAll()}}
function renameRoute(id){{const r=routes.find(x=>x.id===id);if(!r)return;const n=prompt('Tourname',r.name);if(n&&n.trim()){{r.name=n.trim();renderAll()}}}}
function optimizeRoute(id){{const r=routes.find(x=>x.id===id);if(!r||r.id==='UNPLANNED')return;const src=r.items.length&&new Set(r.items.map(x=>x.Quelle)).size===1?r.items[0].Quelle:'MIX';r.src=src;r.items=nearestOrder(r.items,src);renderAll()}}
function optimizeAll(){{routes.filter(r=>r.id!=='UNPLANNED').forEach(r=>{{const src=r.items.length&&new Set(r.items.map(x=>x.Quelle)).size===1?r.items[0].Quelle:'MIX';r.src=src;r.items=nearestOrder(r.items,src)}});renderAll()}}
function clearPlan(){{if(!confirm('Aktuelle Planung leeren?'))return;const all=routes.flatMap(r=>r.items),seen=new Set(),uniq=[];for(const c of all)if(!seen.has(c.uid)){{seen.add(c.uid);uniq.push(c)}}routes=[{{id:'UNPLANNED',name:'NICHT EINGEPLANT',src:'MIX',items:uniq}}];renderAll()}}
function renderMetrics(){{const rs=routes.filter(r=>r.id!=='UNPLANNED'),un=routes.find(r=>r.id==='UNPLANNED');const planned=rs.reduce((s,r)=>s+r.items.length,0),km=rs.reduce((s,r)=>s+routeKm(r.items,r.src),0);document.getElementById('mCustomers').textContent=selected.length;document.getElementById('mRoutes').textContent=rs.length;document.getElementById('mPlanned').textContent=planned;document.getElementById('mUnplanned').textContent=un?un.items.length:0;document.getElementById('mKm').textContent=km.toFixed(0)+' km'}}
function renderTable(){{const tb=document.getElementById('tbody');let html='';routes.forEach(r=>r.items.forEach((c,i)=>html+=`<tr><td>${{esc(r.name)}}</td><td>${{i+1}}</td><td>${{esc(c.Quelle)}}</td><td>${{c.CSB??''}}</td><td>${{c.SAP??''}}</td><td>${{esc(c.Name)}}</td><td>${{esc(c.Plz)}}</td><td>${{esc(c.Ort)}}</td><td>${{esc(c.old||'')}}</td></tr>`));tb.innerHTML=html}}
function renderMap(){{
  const svg=document.getElementById('geoSvg'),pts=routes.filter(r=>r.id!=='UNPLANNED').flatMap(r=>r.items.map(c=>({{c,r}}))).filter(x=>Number.isFinite(x.c.lat)&&Number.isFinite(x.c.lon)); if(!pts.length){{svg.innerHTML='<text x="40" y="60" fill="#aaa">Keine Geo-Daten vorhanden</text>';return}}
  const lats=pts.map(x=>x.c.lat),lons=pts.map(x=>x.c.lon),minLa=Math.min(...lats),maxLa=Math.max(...lats),minLo=Math.min(...lons),maxLo=Math.max(...lons),pad=45,W=1200,H=640; const X=lo=>pad+(lo-minLo)/Math.max(.0001,maxLo-minLo)*(W-2*pad),Y=la=>H-pad-(la-minLa)/Math.max(.0001,maxLa-minLa)*(H-2*pad);
  let h='';const rs=routes.filter(r=>r.id!=='UNPLANNED');rs.forEach((r,ri)=>{{const col=COLORS[ri%COLORS.length],p=r.items.filter(c=>Number.isFinite(c.lat)&&Number.isFinite(c.lon));if(p.length>1)h+=`<polyline points="${{p.map(c=>X(c.lon)+','+Y(c.lat)).join(' ')}}" fill="none" stroke="${{col}}" stroke-width="3" opacity=".75"/>`;p.forEach((c,i)=>h+=`<circle cx="${{X(c.lon)}}" cy="${{Y(c.lat)}}" r="6" fill="${{col}}"><title>${{esc(r.name+' '+(i+1)+' · '+c.SAP+' · '+c.Ort+' · '+c.Name)}}</title></circle>`);}});svg.innerHTML=h;document.getElementById('legend').innerHTML=rs.map((r,i)=>`<span><i style="background:${{COLORS[i%COLORS.length]}}"></i>${{esc(r.name)}}</span>`).join('')
}}
function showTab(name,btn){{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tabbtn').forEach(x=>x.classList.remove('active'));document.getElementById('tab-'+name).classList.add('active');btn.classList.add('active');if(name==='map')renderMap();if(name==='table')renderTable()}}
function exportCsv(){{let rows=[['Neue Tour','Reihenfolge','Quelle','CSB','SAP','Name','Strasse','PLZ','Ort','Bisherige Touren']];routes.forEach(r=>r.items.forEach((c,i)=>rows.push([r.name,i+1,c.Quelle,c.CSB??'',c.SAP??'',c.Name,c.Strasse??'',c.Plz,c.Ort,c.old??''])));const csv='\ufeff'+rows.map(r=>r.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(';')).join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));a.download='Feiertagstouren.csv';a.click();URL.revokeObjectURL(a.href)}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]))}}
init();
</script>
</body></html>'''


st.title("🧭 Feiertags-Tourenplaner – HTML Generator")
st.write("Aktuelle Quelldatei hochladen → Daten der ersten vier Blätter werden übernommen → fertige **HTML-Datei herunterladen**. Die Tourenberechnung und das Verschieben der Kunden erfolgen anschließend direkt in der HTML.")

uploaded = st.file_uploader("Aktuelle Quelldatei hochladen", type=["xlsx"])

with st.expander("Startpunkte / Depot-PLZ für die spätere Reihenfolge", expanded=False):
    depot_plz: Dict[str, str] = {}
    cols = st.columns(4)
    for i, s in enumerate(SHEETS):
        depot_plz[s] = cols[i].text_input(s, value=DEFAULT_DEPOT_PLZ[s], max_chars=5)

if uploaded is None:
    st.info("Bitte die aktuelle Quelldatei hochladen. Erwartet werden die Blätter DIREKT, MK, HUPA_NMS und HUPA_MALCHOW.")
    st.stop()

if pgeocode is None:
    st.error("`pgeocode` fehlt. Bitte requirements.txt verwenden bzw. `pgeocode` installieren.")
    st.stop()

try:
    raw = read_source(uploaded.getvalue())
    with st.spinner("PLZ-Koordinaten werden vorbereitet …"):
        enriched = enrich_geo(raw)
        depots = {s: {"plz": depot_plz[s], **coord_for_plz(depot_plz[s])} for s in SHEETS}
except Exception as exc:
    st.error(f"Datei konnte nicht verarbeitet werden: {exc}")
    st.stop()

geo_ok = int(enriched[["lat", "lon"]].notna().all(axis=1).sum())
geo_bad = int(len(enriched) - geo_ok)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kunden", len(enriched))
c2.metric("Geo erkannt", geo_ok)
c3.metric("ohne Geo", geo_bad)
c4.metric("Blätter", 4)

if geo_bad:
    st.warning(f"{geo_bad} Datensätze konnten über die PLZ nicht geokodiert werden. Sie erscheinen in der HTML zunächst unter ›Nicht eingeplant‹.")

payload = make_payload(enriched)
html = build_html(payload, depots, uploaded.name)

st.success("HTML ist erstellt. Nach dem Download wird keine Excel-Datei mehr benötigt – die Kundendaten sind in der HTML eingebettet.")
st.download_button(
    "⬇ Feiertags-Tourenplaner.html herunterladen",
    data=html.encode("utf-8"),
    file_name="Feiertags_Tourenplaner.html",
    mime="text/html",
    use_container_width=True,
    type="primary",
)

st.caption("In der HTML: Liefertage/Bereiche wählen → Touren berechnen → Kunden per Drag & Drop verschieben → einzelne oder alle Reihenfolgen optimieren → CSV exportieren oder drucken.")

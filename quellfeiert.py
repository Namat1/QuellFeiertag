from __future__ import annotations

import io
import json
import math
from datetime import datetime
from html import escape
from typing import List

import pandas as pd

SHEETS = ["DIREKT", "MK", "HUPA_NMS", "HUPA_MALCHOW"]
DAY_COLUMNS = ["Mo", "Die", "Mitt", "Don", "Fr", "Sam"]
DAY_LABELS = {"Mo": "Montag", "Die": "Dienstag", "Mitt": "Mittwoch", "Don": "Donnerstag", "Fr": "Freitag", "Sam": "Samstag"}
DEFAULT_DEPOT_PLZ = {"DIREKT": "24539", "MK": "24539", "HUPA_NMS": "24539", "HUPA_MALCHOW": "17213"}


def clean_plz(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype("Int64").astype("string")
    return s.str.zfill(5)


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
        df["source_row"] = range(1, len(df) + 1)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["SAP"] = pd.to_numeric(out["SAP"], errors="coerce").astype("Int64")
    out["CSB"] = pd.to_numeric(out["CSB"], errors="coerce").astype("Int64")
    out["Plz"] = clean_plz(out["Plz"])
    for d in DAY_COLUMNS:
        out[d] = pd.to_numeric(out[d], errors="coerce").astype("Int64")

    out = out.dropna(subset=["SAP", "Name", "Plz"]).copy()
    out["uid"] = out.apply(
        lambda r: f"{r['Quelle']}::{int(r['SAP'])}::{int(r['CSB']) if pd.notna(r['CSB']) else 'x'}::{int(r['source_row'])}",
        axis=1,
    )
    return out


def geocode_postcodes(postcodes: list[str]) -> pd.DataFrame:
    try:
        import pgeocode
    except ImportError as exc:
        raise RuntimeError("Das Paket 'pgeocode' fehlt. Bitte requirements.txt installieren.") from exc

    vals = sorted({str(x).zfill(5) for x in postcodes if x and str(x) != "<NA>"})
    if not vals:
        return pd.DataFrame(columns=["Plz", "lat", "lon"])

    geo = pgeocode.Nominatim("de")
    r = geo.query_postal_code(vals)
    if isinstance(r, pd.Series):
        r = r.to_frame().T
    return pd.DataFrame({
        "Plz": pd.Series(vals, dtype="string"),
        "lat": pd.to_numeric(r["latitude"], errors="coerce").to_numpy(),
        "lon": pd.to_numeric(r["longitude"], errors="coerce").to_numpy(),
    })


def enrich_geo(df: pd.DataFrame) -> pd.DataFrame:
    coords = geocode_postcodes(df["Plz"].dropna().astype(str).unique().tolist())
    return df.merge(coords, on="Plz", how="left")


def coord_for_plz(plz: str) -> dict:
    x = geocode_postcodes([str(plz).zfill(5)])
    if x.empty or x[["lat", "lon"]].isna().any(axis=None):
        return {"lat": None, "lon": None}
    return {"lat": float(x.iloc[0]["lat"]), "lon": float(x.iloc[0]["lon"])}


def json_ready(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def make_payload(df: pd.DataFrame) -> list[dict]:
    keep = ["uid", "Quelle", "source_row", "CSB", "SAP", "Name", "Strasse", "Plz", "Ort", *DAY_COLUMNS, "lat", "lon"]
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
:root{{--bg:#f4f4f6;--surface:#ffffff;--surface2:#f8f8fa;--ink:#202126;--muted:#737680;--line:#dedfe4;--line2:#ececf0;--purple:#7256a8;--purpleSoft:#eee9f8;--green:#287a4b;--greenBg:#e8f5ed;--amber:#9a6500;--amberBg:#fff4d6;--red:#a43b3b;--redBg:#fdeaea;--shadow:0 8px 24px rgba(27,28,33,.07)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}}
button,input{{font:inherit}}button{{cursor:pointer}}
.top{{position:sticky;top:0;z-index:30;background:rgba(244,244,246,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}
.topin{{max-width:1900px;margin:auto;padding:16px 22px 14px}}.brandrow{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}}h1{{font-size:24px;line-height:1.1;margin:0;font-weight:800;letter-spacing:-.02em}}.subtitle{{color:var(--muted);font-size:12px;margin-top:5px}}.filepill{{white-space:nowrap;background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--muted);font-size:12px}}
.controlbar{{display:grid;grid-template-columns:1.05fr 1.6fr minmax(190px,.8fr) auto;gap:10px;align-items:end;margin-top:14px}}.control{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:9px 10px}}.label{{display:block;font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:7px}}.chips{{display:flex;gap:6px;flex-wrap:wrap}}.chip{{display:flex;align-items:center;gap:5px;padding:5px 8px;background:var(--surface2);border:1px solid var(--line2);border-radius:999px;font-size:12px}}.chip input{{accent-color:var(--purple)}}
.dayseg{{display:flex;gap:5px}}.daybtn{{border:1px solid var(--line);background:var(--surface2);border-radius:8px;padding:6px 9px;color:#555861;font-weight:700}}.daybtn.active{{background:var(--purple);border-color:var(--purple);color:white}}.search{{width:100%;border:1px solid var(--line);background:var(--surface2);border-radius:8px;padding:7px 9px;outline:none}}.search:focus{{border-color:#a995cd;box-shadow:0 0 0 3px var(--purpleSoft)}}
.actions{{display:flex;gap:7px;align-items:center}}.btn{{border:1px solid var(--line);background:var(--surface);border-radius:9px;padding:8px 11px;color:#35373e;font-weight:700}}.btn:hover{{background:#fafafa}}.btn.primary{{background:var(--purple);border-color:var(--purple);color:#fff}}.btn.danger{{color:var(--red);background:#fff}}
main{{max-width:1900px;margin:auto;padding:16px 22px 50px}}.notice{{background:#fff;border:1px solid var(--line);border-left:4px solid var(--purple);border-radius:10px;padding:10px 12px;margin-bottom:12px;color:#575a63;font-size:12px}}.notice b{{color:var(--ink)}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(125px,1fr));gap:9px;margin-bottom:12px}}.metric{{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:9px 11px}}.metric .v{{font-size:19px;font-weight:800;letter-spacing:-.02em}}.metric .k{{font-size:11px;color:var(--muted);margin-top:1px}}
.toolbar{{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:7px 0 12px}}.spacer{{flex:1}}.toggle{{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}}.toggle input{{accent-color:var(--purple)}}
.board{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:10px;align-items:start}}.route{{background:var(--surface);border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 1px 0 rgba(0,0,0,.02)}}.route.changed{{box-shadow:0 0 0 2px rgba(114,86,168,.16),var(--shadow)}}.route.parking{{border-style:dashed;background:#fffafa}}.routehead{{padding:10px 11px;border-bottom:1px solid var(--line2);display:flex;justify-content:space-between;align-items:flex-start;gap:10px;background:linear-gradient(180deg,#fff,#fbfbfc)}}.rtitle{{font-size:15px;font-weight:850;display:flex;align-items:center;gap:7px}}.sourcebadge{{font-size:9px;font-weight:800;color:#615f67;background:#efeff2;border-radius:999px;padding:3px 6px}}.rmeta{{font-size:11px;color:var(--muted);margin-top:2px}}.rdelta{{font-weight:800}}.rdelta.plus{{color:var(--green)}}.rdelta.minus{{color:var(--red)}}
.dropzone{{min-height:74px;padding:5px 7px 8px}}.dropzone.over{{background:var(--purpleSoft)}}.cust{{position:relative;background:var(--surface);border:1px solid var(--line2);border-radius:9px;padding:7px 8px;margin:5px 0;display:grid;grid-template-columns:16px 1fr auto;gap:6px;align-items:start;cursor:grab;transition:border-color .12s,box-shadow .12s,transform .12s}}.cust:hover{{border-color:#cfc8dc;box-shadow:0 4px 12px rgba(31,31,38,.06)}}.cust.dragging{{opacity:.35;transform:scale(.99)}}.cust.moved{{border-left:4px solid var(--purple)}}.grip{{color:#b0b1b7;font-weight:900;padding-top:2px;user-select:none}}.cname{{font-weight:760;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.csub{{font-size:10px;color:var(--muted);margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.original{{font-size:10px;color:#705b96;margin-top:3px}}.remove{{border:0;background:transparent;color:#b2b3b8;border-radius:6px;padding:2px 5px;font-size:15px;line-height:1}}.remove:hover{{color:var(--red);background:var(--redBg)}}
.fitrow{{display:flex;gap:4px;align-items:center;flex-wrap:wrap;margin-top:5px}}.fit{{display:inline-flex;align-items:center;border-radius:999px;padding:2px 6px;font-size:9px;font-weight:850}}.fit.good{{background:var(--greenBg);color:var(--green)}}.fit.ok{{background:#edf3e7;color:#57792f}}.fit.warn{{background:var(--amberBg);color:var(--amber)}}.fit.bad{{background:var(--redBg);color:var(--red)}}.fit.unknown{{background:#ededf0;color:#737680}}.fitdetail{{font-size:9px;color:var(--muted)}}.crosssource{{font-size:9px;color:var(--amber);font-weight:800}}
.empty{{color:#a0a2aa;font-size:11px;text-align:center;padding:18px 8px}}.hidden{{display:none!important}}
.drawer{{position:fixed;right:18px;bottom:18px;z-index:60;width:min(420px,calc(100vw - 36px));background:#26232b;color:#fff;border-radius:14px;padding:13px 14px;box-shadow:0 18px 50px rgba(0,0,0,.24);transform:translateY(140%);transition:transform .2s ease}}.drawer.show{{transform:translateY(0)}}.drawer .dtop{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.drawer .dbadge{{font-weight:850}}.drawer .small{{font-size:11px;color:#c7c1cf;margin-top:5px}}.drawer.good{{border-left:5px solid #4bb47a}}.drawer.ok{{border-left:5px solid #89a85b}}.drawer.warn{{border-left:5px solid #e4aa3d}}.drawer.bad{{border-left:5px solid #e36b6b}}
.panel{{display:none;margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:12px;overflow:hidden}}.panel.open{{display:block}}table{{width:100%;border-collapse:collapse}}th,td{{font-size:11px;text-align:left;padding:7px 8px;border-bottom:1px solid var(--line2)}}th{{background:#fafafd;color:#666871;position:sticky;top:0}}.tablewrap{{max-height:560px;overflow:auto}}
@media(max-width:1050px){{.controlbar{{grid-template-columns:1fr 1fr}}.metrics{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:650px){{.topin,main{{padding-left:10px;padding-right:10px}}.brandrow{{display:block}}.filepill{{display:inline-block;margin-top:8px}}.controlbar{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr 1fr}}.board{{grid-template-columns:1fr}}}}
@media print{{.top,.toolbar,.notice,.remove,.drawer{{display:none!important}}body{{background:#fff}}main{{max-width:none;padding:0}}.route{{break-inside:avoid;box-shadow:none}}}}
</style>
</head>
<body>
<div class="top"><div class="topin">
  <div class="brandrow"><div><h1>Feiertags-Tourenplaner</h1><div class="subtitle">Bestehende Touren bleiben erhalten. Du verschiebst – die App bewertet die geografische Passung.</div></div><div class="filepill">{src} · erzeugt {generated}</div></div>
  <div class="controlbar">
    <div class="control"><span class="label">Liefertag</span><div class="dayseg" id="days"></div></div>
    <div class="control"><span class="label">Bereiche</span><div class="chips" id="sources"></div></div>
    <div class="control"><span class="label">Suche</span><input id="search" class="search" placeholder="SAP, Name, Ort, Tour …"></div>
    <div class="actions"><button class="btn primary" onclick="loadOriginalPlan()">Touren laden</button><button class="btn" onclick="resetPlan()">Zurücksetzen</button></div>
  </div>
</div></div>
<main>
  <div class="notice"><b>Bewertung:</b> Nur verschobene Kunden werden bewertet. Entscheidend sind Nähe zu Kunden der Ziel-Tour und die geschätzten zusätzlichen Geo-km. Die Rechnung nutzt PLZ-Mittelpunkte/Luftlinie – sie ersetzt kein LKW-Straßenrouting.</div>
  <div class="metrics">
    <div class="metric"><div class="v" id="mRoutes">–</div><div class="k">bestehende Touren</div></div>
    <div class="metric"><div class="v" id="mCustomers">–</div><div class="k">Kunden</div></div>
    <div class="metric"><div class="v" id="mMoved">–</div><div class="k">verschoben</div></div>
    <div class="metric"><div class="v" id="mGood">–</div><div class="k">davon passend</div></div>
    <div class="metric"><div class="v" id="mOut">–</div><div class="k">ausgeplant</div></div>
  </div>
  <div class="toolbar">
    <label class="toggle"><input type="checkbox" id="onlyChanged" onchange="renderAll()"> nur geänderte Touren</label>
    <label class="toggle"><input type="checkbox" id="onlyMoved" onchange="renderAll()"> nur verschobene Kunden</label>
    <div class="spacer"></div>
    <button class="btn" onclick="toggleTable()">Änderungsliste</button>
    <button class="btn" onclick="downloadCSV()">CSV exportieren</button>
    <button class="btn" onclick="window.print()">Drucken</button>
  </div>
  <div class="board" id="board"></div>
  <div class="panel" id="changePanel"><div class="tablewrap"><table><thead><tr><th>SAP</th><th>Kunde</th><th>Von</th><th>Nach</th><th>Bewertung</th><th>Nächster Kunde</th><th>Distanz</th><th>+ Geo-km</th></tr></thead><tbody id="changeBody"></tbody></table></div></div>
</main>
<div class="drawer" id="drawer"><div class="dtop"><div class="dbadge" id="drawerTitle"></div><button onclick="hideDrawer()" style="border:0;background:transparent;color:#fff;font-size:18px">×</button></div><div class="small" id="drawerText"></div></div>
<script>
const CUSTOMERS={data_json};
const DEPOTS={dep_json};
const DAYS={{Mo:'Montag',Die:'Dienstag',Mitt:'Mittwoch',Don:'Donnerstag',Fr:'Freitag',Sam:'Samstag'}};
const SOURCES=['DIREKT','MK','HUPA_NMS','HUPA_MALCHOW'];
let activeDay='Mo', routes=[], selected=[], dragUid=null, originalSnapshot={{}}, searchText='';
const esc=s=>String(s??'').replace(/[&<>\"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[m]));
function hav(a,b){{if(!a||!b||!Number.isFinite(a.lat)||!Number.isFinite(a.lon)||!Number.isFinite(b.lat)||!Number.isFinite(b.lon))return Infinity;const R=6371,toR=x=>x*Math.PI/180,dla=toR(b.lat-a.lat),dlo=toR(b.lon-a.lon),la1=toR(a.lat),la2=toR(b.lat);const h=Math.sin(dla/2)**2+Math.cos(la1)*Math.cos(la2)*Math.sin(dlo/2)**2;return 2*R*Math.asin(Math.sqrt(h))}}
function median(a){{const x=a.filter(Number.isFinite).sort((p,q)=>p-q);if(!x.length)return NaN;const m=Math.floor(x.length/2);return x.length%2?x[m]:(x[m-1]+x[m])/2}}
function depotFor(src){{const d=DEPOTS[src];return d&&Number.isFinite(d.lat)&&Number.isFinite(d.lon)?d:null}}
function greedyOrder(arr,src){{if(arr.length<2)return arr.slice();const left=arr.slice(),dep=depotFor(src);let cur=dep||left[0],out=[];while(left.length){{let bi=0,bd=Infinity;left.forEach((x,i)=>{{const d=hav(cur,x);if(d<bd){{bd=d;bi=i}}}});const n=left.splice(bi,1)[0];out.push(n);cur=n}}return out}}
function openKm(arr,src){{const x=greedyOrder(arr.filter(c=>Number.isFinite(c.lat)&&Number.isFinite(c.lon)),src);if(!x.length)return 0;let km=0,dep=depotFor(src);if(dep)km+=hav(dep,x[0]);for(let i=1;i<x.length;i++)km+=hav(x[i-1],x[i]);return Number.isFinite(km)?km:0}}
function typicalSpacing(peers){{const p=peers.filter(c=>Number.isFinite(c.lat)&&Number.isFinite(c.lon));if(p.length<2)return NaN;const ds=p.map((c,i)=>Math.min(...p.filter((_,j)=>j!==i).map(o=>hav(c,o))));return median(ds)}}
function bestInsertionExtra(c,peers,src){{const p=peers.filter(x=>Number.isFinite(x.lat)&&Number.isFinite(x.lon));if(!Number.isFinite(c.lat)||!Number.isFinite(c.lon)||!p.length)return NaN;const base=openKm(p,src),ord=greedyOrder(p,src);let best=Infinity;for(let i=0;i<=ord.length;i++){{const cand=ord.slice();cand.splice(i,0,c);best=Math.min(best,openKm(cand,src)-base)}}return Math.max(0,best)}}
function getRoute(id){{return routes.find(r=>r.id===id)}}
function currentRouteOf(uid){{return routes.find(r=>r.items.some(c=>c.uid===uid))}}
function originalRouteId(c){{return c.originalRouteId}}
function fitFor(c,r){{if(!r||r.parking)return {{key:'unknown',label:'ausgeplant',nearest:NaN,nearName:'',extra:NaN,note:''}};const peers=r.items.filter(x=>x.uid!==c.uid&&Number.isFinite(x.lat)&&Number.isFinite(x.lon));if(!Number.isFinite(c.lat)||!Number.isFinite(c.lon)||!peers.length)return {{key:'unknown',label:'nicht bewertbar',nearest:NaN,nearName:'',extra:NaN,note:'zu wenig Geo-Daten'}};let nearest=Infinity,near=null;peers.forEach(p=>{{const d=hav(c,p);if(d<nearest){{nearest=d;near=p}}}});const typ=typicalSpacing(peers),extra=bestInsertionExtra(c,peers,r.source);const ref=Number.isFinite(typ)?Math.max(12,typ*1.45):18;let key,label;if((nearest<=14||nearest<=ref*.8)&&(extra<=14||extra<=ref)){{key='good';label='passt sehr gut'}}else if((nearest<=28||nearest<=ref*1.55)&&(extra<=25||extra<=ref*1.8)){{key='ok';label='passt gut'}}else if((nearest<=48||nearest<=ref*2.4)&&(extra<=42||extra<=ref*3)){{key='warn';label='grenzwertig'}}else{{key='bad';label='eher nicht'}}const cross=c.Quelle!==r.source;return {{key,label,nearest,nearName:near?`${{near.SAP}} · ${{near.Ort}}`:'',extra,note:cross?'anderer Bereich':''}}}}
function fmt(x){{return Number.isFinite(x)?x.toFixed(1):'–'}}
function initControls(){{const d=document.getElementById('days');Object.entries(DAYS).forEach(([k,v])=>{{const b=document.createElement('button');b.className='daybtn'+(k===activeDay?' active':'');b.textContent=v.slice(0,2);b.title=v;b.onclick=()=>{{activeDay=k;document.querySelectorAll('.daybtn').forEach(x=>x.classList.remove('active'));b.classList.add('active');loadOriginalPlan()}};d.appendChild(b)}});const s=document.getElementById('sources');SOURCES.forEach(src=>{{const id='src_'+src;const l=document.createElement('label');l.className='chip';l.innerHTML=`<input id="${{id}}" type="checkbox" checked value="${{src}}"><span>${{src.replace('HUPA_','HUPA ')}}</span>`;s.appendChild(l)}});document.getElementById('search').addEventListener('input',e=>{{searchText=e.target.value.trim().toLowerCase();renderBoard()}})}}
function activeSources(){{return SOURCES.filter(s=>document.getElementById('src_'+s)?.checked)}}
function routeId(source,tour){{return source+'::'+String(tour)}}
function loadOriginalPlan(){{const sources=activeSources();selected=CUSTOMERS.filter(c=>sources.includes(c.Quelle)&&c[activeDay]!=null).map(c=>({{...c}}));const map=new Map();selected.forEach(c=>{{const tour=String(c[activeDay]),id=routeId(c.Quelle,tour);c.originalRouteId=id;c.originalTour=tour;if(!map.has(id))map.set(id,{{id,tour,source:c.Quelle,items:[],originalCount:0,parking:false}});map.get(id).items.push(c)}});routes=[...map.values()].sort((a,b)=>Number(a.tour)-Number(b.tour)||a.source.localeCompare(b.source));routes.forEach(r=>{{r.items.sort((a,b)=>(a.source_row??0)-(b.source_row??0));r.originalCount=r.items.length}});routes.push({{id:'PARKING',tour:'AUSGEPLANT',source:'-',items:[],originalCount:0,parking:true}});originalSnapshot={{}};selected.forEach(c=>originalSnapshot[c.uid]=c.originalRouteId);renderAll()}}
function resetPlan(){{if(routes.length&& !confirm('Alle manuellen Änderungen zurücksetzen?'))return;loadOriginalPlan()}}
function routeDelta(r){{if(r.parking)return 0;return r.items.length-r.originalCount}}
function routeChanged(r){{if(r.parking)return r.items.length>0;return routeDelta(r)!==0||r.items.some(c=>c.originalRouteId!==r.id)}}
function customerMatches(c,r){{if(!searchText)return true;return [c.SAP,c.CSB,c.Name,c.Ort,c.Plz,c.Strasse,r.tour,r.source].some(v=>String(v??'').toLowerCase().includes(searchText))}}
function custHtml(c,r){{const moved=!r.parking&&c.originalRouteId!==r.id;const fit=moved?fitFor(c,r):null;const cross=moved&&fit?.note;const visible=customerMatches(c,r);if(!visible)return '';let fitHtml='';if(moved)fitHtml=`<div class="fitrow"><span class="fit ${{fit.key}}">${{fit.label}}</span><span class="fitdetail">${{fmt(fit.nearest)}} km zum nächsten · +${{fmt(fit.extra)}} Geo-km</span>${{cross?'<span class="crosssource">anderer Bereich</span>':''}}</div>`;return `<div class="cust ${{moved?'moved':''}}" draggable="true" data-uid="${{c.uid}}"><div class="grip">⋮⋮</div><div><div class="cname">${{c.SAP}} · ${{esc(c.Name)}}</div><div class="csub">${{esc(c.Plz)}} ${{esc(c.Ort)}} · ${{esc(c.Strasse||'')}}</div>${{moved?`<div class="original">von Tour ${{esc(c.originalTour)}} · ${{esc(c.Quelle)}}</div>`:''}}${{fitHtml}}</div><button class="remove" title="Ausplanen" onclick="event.stopPropagation();moveUid('${{c.uid}}','PARKING',null)">×</button></div>`}}
function renderBoard(){{const b=document.getElementById('board');const onlyChanged=document.getElementById('onlyChanged').checked,onlyMoved=document.getElementById('onlyMoved').checked;b.innerHTML='';routes.forEach(r=>{{if(onlyChanged&&!routeChanged(r))return;let cards=r.items.map(c=>{{if(onlyMoved&&c.originalRouteId===r.id&&!r.parking)return '';return custHtml(c,r)}}).join('');if(searchText&&!cards&&!r.parking)return;const d=routeDelta(r),delta=!r.parking&&d!==0?`<span class="rdelta ${{d>0?'plus':'minus'}}">${{d>0?'+':''}}${{d}}</span>`:'';const movedIn=r.parking?0:r.items.filter(c=>c.originalRouteId!==r.id).length;const movedOut=r.parking?0:Math.max(0,r.originalCount-r.items.filter(c=>c.originalRouteId===r.id).length);const meta=r.parking?`${{r.items.length}} Kunden entfernt`:`${{r.items.length}} Kunden · ursprünglich ${{r.originalCount}}${{movedIn||movedOut?` · ${{movedIn}} rein / ${{movedOut}} raus`:''}}`;const el=document.createElement('section');el.className='route'+(routeChanged(r)?' changed':'')+(r.parking?' parking':'');el.dataset.route=r.id;el.innerHTML=`<div class="routehead"><div><div class="rtitle">${{r.parking?'AUSGEPLANT':'Tour '+esc(r.tour)}} ${{delta}} ${{r.parking?'':`<span class="sourcebadge">${{esc(r.source.replace('HUPA_','HUPA '))}}</span>`}}</div><div class="rmeta">${{meta}}</div></div></div><div class="dropzone" data-route="${{r.id}}">${{cards||'<div class="empty">Kunden hierher ziehen</div>'}}</div>`;b.appendChild(el)}});bindDnD()}}
function bindDnD(){{document.querySelectorAll('.cust').forEach(el=>{{el.addEventListener('dragstart',e=>{{dragUid=el.dataset.uid;el.classList.add('dragging');e.dataTransfer.effectAllowed='move'}});el.addEventListener('dragend',()=>{{el.classList.remove('dragging');document.querySelectorAll('.dropzone').forEach(z=>z.classList.remove('over'))}});el.addEventListener('dragover',e=>e.preventDefault());el.addEventListener('drop',e=>{{e.preventDefault();e.stopPropagation();moveUid(dragUid,el.closest('.dropzone').dataset.route,el.dataset.uid)}})}});document.querySelectorAll('.dropzone').forEach(z=>{{z.addEventListener('dragover',e=>{{e.preventDefault();z.classList.add('over')}});z.addEventListener('dragleave',()=>z.classList.remove('over'));z.addEventListener('drop',e=>{{if(e.target.closest('.cust'))return;e.preventDefault();z.classList.remove('over');moveUid(dragUid,z.dataset.route,null)}})}})}}
function findRemove(uid){{for(const r of routes){{const i=r.items.findIndex(c=>c.uid===uid);if(i>=0)return {{c:r.items.splice(i,1)[0],from:r}}}}return null}}
function moveUid(uid,toId,beforeUid){{if(!uid)return;const found=findRemove(uid),to=getRoute(toId);if(!found||!to)return;let ix=beforeUid?to.items.findIndex(x=>x.uid===beforeUid):-1;if(ix<0)to.items.push(found.c);else to.items.splice(ix,0,found.c);renderAll();if(!to.parking&&found.c.originalRouteId!==to.id)showMoveResult(found.c,to);else if(to.parking)showParking(found.c)}}
function showMoveResult(c,r){{const f=fitFor(c,r),d=document.getElementById('drawer');d.className='drawer show '+f.key;document.getElementById('drawerTitle').textContent=`Tour ${{r.tour}}: ${{f.label}}`;document.getElementById('drawerText').textContent=`${{c.SAP}} · ${{c.Ort}} → nächster Kunde: ${{f.nearName||'–'}} (${{fmt(f.nearest)}} km), geschätzter Zusatz: +${{fmt(f.extra)}} Geo-km.${{f.note?' Hinweis: '+f.note+'.':''}}`;clearTimeout(window.__dt);window.__dt=setTimeout(hideDrawer,6500)}}
function showParking(c){{const d=document.getElementById('drawer');d.className='drawer show warn';document.getElementById('drawerTitle').textContent='Kunde ausgeplant';document.getElementById('drawerText').textContent=`${{c.SAP}} · ${{c.Name}} liegt jetzt im Bereich AUSGEPLANT und kann jederzeit wieder in eine Tour gezogen werden.`;clearTimeout(window.__dt);window.__dt=setTimeout(hideDrawer,4500)}}
function hideDrawer(){{document.getElementById('drawer').className='drawer'}}
function renderMetrics(){{const rs=routes.filter(r=>!r.parking),park=routes.find(r=>r.parking);let moved=0,good=0;rs.forEach(r=>r.items.forEach(c=>{{if(c.originalRouteId!==r.id){{moved++;const f=fitFor(c,r);if(f.key==='good'||f.key==='ok')good++}}}}));document.getElementById('mRoutes').textContent=rs.length;document.getElementById('mCustomers').textContent=selected.length;document.getElementById('mMoved').textContent=moved;document.getElementById('mGood').textContent=good;document.getElementById('mOut').textContent=park?.items.length||0}}
function renderChanges(){{const tb=document.getElementById('changeBody');let h='';routes.forEach(r=>r.items.forEach(c=>{{if(r.parking){{h+=`<tr><td>${{c.SAP}}</td><td>${{esc(c.Name)}}</td><td>${{esc(c.originalTour)}}</td><td>AUSGEPLANT</td><td>–</td><td>–</td><td>–</td><td>–</td></tr>`;return}}if(c.originalRouteId!==r.id){{const f=fitFor(c,r);h+=`<tr><td>${{c.SAP}}</td><td>${{esc(c.Name)}}</td><td>${{esc(c.originalTour)}}</td><td>${{esc(r.tour)}}</td><td>${{esc(f.label)}}</td><td>${{esc(f.nearName)}}</td><td>${{fmt(f.nearest)}} km</td><td>+${{fmt(f.extra)}} km</td></tr>`}}}}));tb.innerHTML=h||'<tr><td colspan="8" style="color:#888">Noch keine Änderungen.</td></tr>'}}
function toggleTable(){{const p=document.getElementById('changePanel');p.classList.toggle('open');if(p.classList.contains('open'))renderChanges()}}
function renderAll(){{renderBoard();renderMetrics();if(document.getElementById('changePanel').classList.contains('open'))renderChanges()}}
function csvCell(v){{const s=String(v??'');const q=s.includes(';')||s.includes('\"')||s.indexOf(String.fromCharCode(10))>=0||s.indexOf(String.fromCharCode(13))>=0;return q?'\"'+s.replace(/\"/g,'\"\"')+'\"':s}}
function downloadCSV(){{const rows=[['Tag','Bereich','Tour','Pos','CSB','SAP','Name','PLZ','Ort','Straße','Originaltour','Status','Bewertung','Nächster Kunde','Distanz km','Zusatz Geo-km']];routes.forEach(r=>r.items.forEach((c,i)=>{{const moved=!r.parking&&c.originalRouteId!==r.id,f=moved?fitFor(c,r):null;rows.push([DAYS[activeDay],c.Quelle,r.parking?'AUSGEPLANT':r.tour,i+1,c.CSB??'',c.SAP??'',c.Name,c.Plz,c.Ort,c.Strasse||'',c.originalTour,r.parking?'ausgeplant':moved?'verschoben':'unverändert',f?.label||'',f?.nearName||'',Number.isFinite(f?.nearest)?f.nearest.toFixed(1):'',Number.isFinite(f?.extra)?f.extra.toFixed(1):''])}}));const blob=new Blob(['\ufeff'+rows.map(r=>r.map(csvCell).join(';')).join(String.fromCharCode(13,10))],{{type:'text/csv;charset=utf-8'}}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='Feiertagsplanung_'+DAYS[activeDay]+'.csv';a.click();URL.revokeObjectURL(a.href)}}
initControls();loadOriginalPlan();
</script>
</body></html>'''


def main():
    import streamlit as st

    st.set_page_config(page_title="Feiertags-Tourenplaner – HTML Generator", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
        .block-container{max-width:1150px;padding-top:1.5rem}
        [data-testid="stFileUploader"]{background:#fafafa;border-radius:12px;padding:8px}
        .small{color:#777;font-size:.88rem}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Feiertags-Tourenplaner – HTML Generator")
    st.caption("Aktuelle Excel hochladen → Kunden geocodieren → eigenständige HTML herunterladen. In der HTML bleiben die vorhandenen Touren zunächst unverändert.")

    upload = st.file_uploader("Aktuelle Quelldatei (.xlsx)", type=["xlsx"])
    if upload is None:
        st.info("Benötigt werden die ersten vier Blätter: DIREKT, MK, HUPA_NMS und HUPA_MALCHOW.")
        return

    file_bytes = upload.getvalue()
    try:
        df = read_source(file_bytes)
    except Exception as exc:
        st.error(f"Datei konnte nicht gelesen werden: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kunden", f"{len(df):,}".replace(",", "."))
    c2.metric("DIREKT", int((df["Quelle"] == "DIREKT").sum()))
    c3.metric("HUPA NMS", int((df["Quelle"] == "HUPA_NMS").sum()))
    c4.metric("HUPA Malchow", int((df["Quelle"] == "HUPA_MALCHOW").sum()))

    st.subheader("Depot-PLZ für die Geo-Bewertung")
    cols = st.columns(4)
    depot_plz = {}
    labels = {"DIREKT": "DIREKT", "MK": "MK", "HUPA_NMS": "HUPA NMS", "HUPA_MALCHOW": "HUPA Malchow"}
    for col, src in zip(cols, SHEETS):
        depot_plz[src] = col.text_input(labels[src], value=DEFAULT_DEPOT_PLZ[src], max_chars=5)

    st.markdown("<div class='small'>Die Depot-PLZ beeinflusst nur die Schätzung der zusätzlichen Geo-km. Die Bewertung nutzt PLZ-Mittelpunkte und keine echten Straßenkilometer.</div>", unsafe_allow_html=True)

    if st.button("HTML erzeugen", type="primary", use_container_width=True):
        try:
            with st.spinner("PLZ-Koordinaten werden ergänzt …"):
                geo_df = enrich_geo(df)
                depots = {}
                for src in SHEETS:
                    c = coord_for_plz(depot_plz[src])
                    depots[src] = {"plz": str(depot_plz[src]).zfill(5), **c}
                html = build_html(make_payload(geo_df), depots, upload.name)
            missing_geo = int(geo_df[["lat", "lon"]].isna().any(axis=1).sum())
            st.success(f"HTML erstellt. {len(geo_df)-missing_geo} von {len(geo_df)} Kunden haben Geo-Daten.")
            if missing_geo:
                st.warning(f"Für {missing_geo} Kunden konnte keine PLZ-Koordinate ermittelt werden. Diese Kunden können verschoben werden, erhalten aber ggf. keine Geo-Bewertung.")
            st.download_button(
                "Feiertags_Tourenplaner.html herunterladen",
                data=html.encode("utf-8"),
                file_name="Feiertags_Tourenplaner.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"HTML konnte nicht erzeugt werden: {exc}")


if __name__ == "__main__":
    main()

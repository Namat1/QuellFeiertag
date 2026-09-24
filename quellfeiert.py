from __future__ import annotations

import io
import json
from datetime import datetime
from html import escape
from typing import List

import pandas as pd

SHEETS = ["DIREKT", "MK", "HUPA_NMS", "HUPA_MALCHOW"]
DAY_COLUMNS = ["Mo", "Die", "Mitt", "Don", "Fr", "Sam"]
DAY_LABELS = {
    "Mo": "Montag",
    "Die": "Dienstag",
    "Mitt": "Mittwoch",
    "Don": "Donnerstag",
    "Fr": "Freitag",
    "Sam": "Samstag",
}


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

    return pd.DataFrame(
        {
            "Plz": pd.Series(vals, dtype="string"),
            "lat": pd.to_numeric(r["latitude"], errors="coerce").to_numpy(),
            "lon": pd.to_numeric(r["longitude"], errors="coerce").to_numpy(),
        }
    )


def enrich_geo(df: pd.DataFrame) -> pd.DataFrame:
    coords = geocode_postcodes(df["Plz"].dropna().astype(str).unique().tolist())
    return df.merge(coords, on="Plz", how="left")


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
    keep = [
        "uid", "Quelle", "source_row", "CSB", "SAP", "Name", "Strasse", "Plz", "Ort",
        *DAY_COLUMNS, "lat", "lon",
    ]
    rows = []
    for rec in df[keep].to_dict("records"):
        rows.append({k: json_ready(v) for k, v in rec.items()})
    return rows


def build_html(customers: list[dict], source_name: str) -> str:
    data_json = json.dumps(customers, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    src = escape(source_name)
    generated = datetime.now().strftime("%d.%m.%Y %H:%M")

    return f'''<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feiertags-Wochenplaner V5</title>
<style>
:root{{--bg:#f3f4f6;--surface:#fff;--surface2:#f8f9fb;--ink:#22242a;--muted:#727680;--line:#dfe1e6;--line2:#eceef2;--accent:#6f54a6;--accent-soft:#eeeaf7;--good:#247348;--good-bg:#e8f5ed;--ok:#5e6f32;--ok-bg:#eff4df;--warn:#946200;--warn-bg:#fff3d2;--bad:#a03b3b;--bad-bg:#fde9e9;--blue:#46627d;--blue-bg:#eaf0f5}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}}
button,input,select{{font:inherit}}
button{{cursor:pointer}}
.top{{position:sticky;top:0;z-index:50;background:rgba(243,244,246,.96);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}}
.topin{{max-width:2400px;margin:auto;padding:13px 18px 11px}}
.headrow{{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}}
h1{{font-size:22px;line-height:1.1;margin:0;font-weight:850;letter-spacing:-.02em}}
.sub{{color:var(--muted);font-size:11px;margin-top:4px}}
.filepill{{white-space:nowrap;background:#fff;border:1px solid var(--line);border-radius:999px;padding:6px 9px;color:var(--muted);font-size:11px}}
.controls{{display:grid;grid-template-columns:1.4fr minmax(230px,.8fr) auto;gap:9px;margin-top:11px;align-items:end}}
.ctrl{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px 9px}}
.label{{display:block;font-size:10px;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}}
.chips{{display:flex;gap:5px;flex-wrap:wrap}}
.chip{{display:flex;align-items:center;gap:5px;padding:4px 7px;background:var(--surface2);border:1px solid var(--line2);border-radius:999px;font-size:11px}}
.chip input{{accent-color:var(--accent)}}
.search{{width:100%;border:1px solid var(--line);background:var(--surface2);border-radius:7px;padding:6px 8px;outline:none}}
.search:focus{{border-color:#9e8ac9;box-shadow:0 0 0 3px var(--accent-soft)}}
.actions{{display:flex;gap:6px;align-items:center;flex-wrap:wrap}}
.btn{{border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 9px;color:#373941;font-weight:750}}
.btn:hover{{background:#fafafa}}
.btn.primary{{background:var(--accent);border-color:var(--accent);color:#fff}}
main{{max-width:2400px;margin:auto;padding:13px 18px 45px}}
.notice{{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:9px;padding:9px 11px;margin-bottom:10px;color:#555963;font-size:11px}}
.notice b{{color:var(--ink)}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(115px,1fr));gap:7px;margin-bottom:10px}}
.metric{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:7px 9px}}
.metric .v{{font-size:17px;font-weight:850}}
.metric .k{{font-size:10px;color:var(--muted);margin-top:1px}}
.unplanned{{background:#fff;border:1px dashed #d3b6b6;border-radius:10px;margin-bottom:10px;overflow:hidden}}
.unplanned-head{{display:flex;align-items:center;justify-content:space-between;padding:7px 9px;background:#fffafa;border-bottom:1px dashed #ead6d6}}
.unplanned-title{{font-weight:850;color:#844}}
.unplanned-drop{{min-height:45px;padding:5px;display:flex;gap:5px;flex-wrap:wrap;align-items:flex-start}}
.weekwrap{{overflow-x:auto;padding-bottom:8px}}
.week{{display:grid;grid-template-columns:repeat(6,minmax(270px,1fr));gap:7px;min-width:1660px;align-items:stretch}}
.daycol{{min-width:0;height:calc(100vh - 285px);min-height:470px;display:flex;flex-direction:column}}
.dayhead{{background:#373941;color:white;border-radius:9px 9px 0 0;padding:8px 9px;display:flex;justify-content:space-between;align-items:center;gap:7px;border:1px solid #373941;flex:0 0 auto}}
.dayhead-main{{min-width:0}}
.dayname{{font-size:14px;font-weight:850}}
.daymeta{{font-size:10px;opacity:.78}}
.addtour{{border:1px solid rgba(255,255,255,.3);background:rgba(255,255,255,.10);color:#fff;border-radius:7px;padding:5px 7px;font-size:10px;font-weight:850;white-space:nowrap}}
.addtour:hover{{background:rgba(255,255,255,.19)}}
.daybody{{background:#e9eaed;border:1px solid #d8dadd;border-top:0;border-radius:0 0 9px 9px;padding:5px;min-height:0;overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable;flex:1}}
.daybody::-webkit-scrollbar{{width:9px}}
.daybody::-webkit-scrollbar-track{{background:#e2e3e6}}
.daybody::-webkit-scrollbar-thumb{{background:#b7bac1;border-radius:999px;border:2px solid #e2e3e6}}
.route{{background:#fff;border:1px solid var(--line);border-radius:8px;margin-bottom:5px;overflow:hidden}}
.route.changed{{box-shadow:0 0 0 2px rgba(111,84,166,.16)}}
.routehead{{padding:6px 7px;background:#fafafa;border-bottom:1px solid var(--line2);display:flex;align-items:center;justify-content:space-between;gap:5px}}
.route-right{{display:flex;align-items:center;gap:4px}}
.route-delete{{border:0;background:transparent;color:#a2a4aa;border-radius:5px;padding:2px 5px;font-weight:900;font-size:13px;line-height:1}}
.route-delete:hover{{background:var(--bad-bg);color:var(--bad)}}
.rleft{{min-width:0}}
.rtitle{{font-weight:850;font-size:12px;display:flex;gap:5px;align-items:center;min-width:0}}
.source{{font-size:8px;font-weight:850;color:#62656d;background:#e9e9ec;border-radius:999px;padding:2px 5px;white-space:nowrap}}
.rmeta{{font-size:9px;color:var(--muted);margin-top:1px}}
.delta{{font-size:10px;font-weight:850;padding:2px 5px;border-radius:999px;background:var(--accent-soft);color:var(--accent)}}
.dropzone{{min-height:28px;padding:3px}}
.dropzone.over,.unplanned-drop.over{{outline:2px dashed #9e8ac9;outline-offset:-2px;background:#f5f2fb}}
.cust{{display:grid;grid-template-columns:14px minmax(0,1fr) auto;gap:5px;align-items:start;border:1px solid var(--line2);background:#fff;border-radius:6px;padding:5px;margin:3px 0;cursor:grab}}
.cust:active{{cursor:grabbing}}
.cust.moved{{border-left:3px solid var(--accent);background:#fdfcff}}
.cust.badmove{{border-left-color:var(--bad)}}
.cust.dupe{{box-shadow:inset 0 0 0 1px #d9a7a7}}
.grip{{color:#aaa;font-weight:900;line-height:1.2;user-select:none}}
.cmain{{min-width:0}}
.cname{{font-size:10.5px;font-weight:820;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.csub{{font-size:9px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:1px}}
.orig{{font-size:8.5px;color:#7b648d;margin-top:2px}}
.fitrow{{display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin-top:3px}}
.fit{{font-size:8px;font-weight:900;border-radius:999px;padding:2px 5px;white-space:nowrap}}
.fit.good{{background:var(--good-bg);color:var(--good)}}
.fit.ok{{background:var(--ok-bg);color:var(--ok)}}
.fit.warn{{background:var(--warn-bg);color:var(--warn)}}
.fit.bad{{background:var(--bad-bg);color:var(--bad)}}
.fit.unknown{{background:#eee;color:#666}}
.detail{{font-size:8px;color:var(--muted)}}
.flag{{font-size:8px;font-weight:850;border-radius:999px;padding:2px 5px;background:var(--blue-bg);color:var(--blue)}}
.flag.dupe{{background:var(--bad-bg);color:var(--bad)}}
.remove{{border:0;background:transparent;color:#aaa;font-weight:900;font-size:14px;line-height:1;padding:1px 2px;border-radius:4px}}
.remove:hover{{color:var(--bad);background:var(--bad-bg)}}
.empty{{color:#a0a2a8;font-size:9px;text-align:center;padding:7px 4px}}
.drawer{{position:fixed;right:16px;bottom:16px;z-index:100;max-width:430px;background:#fff;border:1px solid var(--line);border-left:5px solid var(--accent);border-radius:10px;padding:10px 12px;box-shadow:0 12px 35px rgba(0,0,0,.15);display:none}}
.modalback{{position:fixed;inset:0;z-index:150;background:rgba(28,30,35,.44);display:none;align-items:center;justify-content:center;padding:20px}}
.modalback.show{{display:flex}}
.modal{{width:min(420px,100%);background:#fff;border:1px solid var(--line);border-radius:12px;box-shadow:0 18px 55px rgba(0,0,0,.22);padding:14px}}
.modal h2{{font-size:16px;margin:0 0 3px}}
.modal .hint{{font-size:10px;color:var(--muted);margin-bottom:12px}}
.formrow{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.field label{{display:block;font-size:9px;font-weight:850;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.field input,.field select{{width:100%;border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff;outline:none}}
.field input:focus,.field select:focus{{border-color:#9e8ac9;box-shadow:0 0 0 3px var(--accent-soft)}}
.modalactions{{display:flex;justify-content:flex-end;gap:6px;margin-top:12px}}
.modalerror{{min-height:16px;margin-top:6px;color:var(--bad);font-size:10px;font-weight:750}}
.drawer.show{{display:block}}
.drawer.good{{border-left-color:var(--good)}}.drawer.ok{{border-left-color:var(--ok)}}.drawer.warn{{border-left-color:var(--warn)}}.drawer.bad{{border-left-color:var(--bad)}}
.drawer h3{{margin:0 0 3px;font-size:13px}}.drawer p{{margin:0;color:var(--muted);font-size:10px}}
.changes{{margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.changehead{{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border-bottom:1px solid var(--line2)}}
.changetable{{width:100%;border-collapse:collapse;font-size:10px}}
.changetable th,.changetable td{{padding:6px 7px;text-align:left;border-bottom:1px solid var(--line2);white-space:nowrap}}
.changetable th{{background:#fafafa;color:#666}}
@media(max-width:900px){{.controls{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}.daycol{{height:calc(100vh - 365px);min-height:430px}}}}
@media print{{.top,.notice,.metrics,.unplanned,.changes,.remove,.drawer,.addtour,.route-delete,.modalback{{display:none!important}}main{{padding:0}}.weekwrap{{overflow:visible}}.week{{min-width:0;grid-template-columns:repeat(3,1fr);gap:4px}}.daycol{{height:auto;min-height:0}}.daybody{{overflow:visible}}.dayhead{{background:#333!important;-webkit-print-color-adjust:exact;print-color-adjust:exact}}.cust{{break-inside:avoid}}}}
</style>
</head>
<body>
<div class="top"><div class="topin">
  <div class="headrow">
    <div><h1>Feiertags-Wochenplaner</h1><div class="sub">Komplette Woche gleichzeitig · jeder Tag separat scrollbar · neue Touren frei anlegen</div></div>
    <div class="filepill">{src} · erzeugt {generated}</div>
  </div>
  <div class="controls">
    <div class="ctrl"><span class="label">Bereiche</span><div class="chips" id="sourceChips"></div></div>
    <div class="ctrl"><span class="label">Suche</span><input id="search" class="search" placeholder="SAP, CSB, Name, Ort, PLZ oder Tour"></div>
    <div class="actions"><button class="btn" id="resetBtn">Zurücksetzen</button><button class="btn" id="csvBtn">Änderungen CSV</button><button class="btn primary" onclick="window.print()">Drucken</button></div>
  </div>
</div></div>
<main>
  <div class="notice"><b>So arbeitet die Planung:</b> Montag bis Samstag stehen gleichzeitig nebeneinander und jeder Tag scrollt für sich. Bestehende Touren bleiben unverändert. Über <b>+ Tour</b> kannst du pro Tag neue leere Touren anlegen und Kunden hineinziehen.</div>
  <div class="metrics">
    <div class="metric"><div class="v" id="mDeliveries">0</div><div class="k">Lieferungen in der Woche</div></div>
    <div class="metric"><div class="v" id="mRoutes">0</div><div class="k">Touren gesamt</div></div>
    <div class="metric"><div class="v" id="mMoved">0</div><div class="k">verschoben</div></div>
    <div class="metric"><div class="v" id="mGood">0</div><div class="k">davon geografisch passend</div></div>
    <div class="metric"><div class="v" id="mOut">0</div><div class="k">ausgeplant</div></div>
  </div>

  <section class="unplanned">
    <div class="unplanned-head"><div class="unplanned-title">AUSGEPLANT</div><div class="sub">Kunden hierher ziehen – sie können später wieder in jede Tour gezogen werden.</div></div>
    <div id="unplannedDrop" class="unplanned-drop" data-route="UNPLANNED"></div>
  </section>

  <div class="weekwrap"><div id="week" class="week"></div></div>

  <section class="changes">
    <div class="changehead"><strong>Änderungen</strong><span class="sub">Originaltag/-tour → neuer Tag/neue Tour</span></div>
    <div style="overflow:auto"><table class="changetable"><thead><tr><th>SAP</th><th>Kunde</th><th>Original</th><th>Neu</th><th>Bewertung</th><th>Nächster Kunde</th><th>Entfernung</th><th>Hinweis</th></tr></thead><tbody id="changeBody"></tbody></table></div>
  </section>
</main>
<div id="drawer" class="drawer"><h3 id="drawerTitle"></h3><p id="drawerText"></p></div>
<div id="tourModal" class="modalback" aria-hidden="true"><div class="modal" role="dialog" aria-modal="true" aria-labelledby="newTourTitle"><h2 id="newTourTitle">Neue Tour anlegen</h2><div class="hint" id="newTourHint"></div><div class="formrow"><div class="field"><label for="newTourNo">Tournummer / Name</label><input id="newTourNo" autocomplete="off" placeholder="z. B. 4055 oder FT-01"></div><div class="field"><label for="newTourSource">Bereich</label><select id="newTourSource"></select></div></div><div id="newTourError" class="modalerror"></div><div class="modalactions"><button class="btn" id="cancelTourBtn">Abbrechen</button><button class="btn primary" id="createTourBtn">Tour anlegen</button></div></div></div>
<script>
const CUSTOMERS={data_json};
const DAY_ORDER=['Mo','Die','Mitt','Don','Fr','Sam'];
const DAY_LABELS={{Mo:'Montag',Die:'Dienstag',Mitt:'Mittwoch',Don:'Donnerstag',Fr:'Freitag',Sam:'Samstag'}};
const SOURCES=['DIREKT','MK','HUPA_NMS','HUPA_MALCHOW'];
let routes=[];
let assignments=[];
let unplanned=[];
let searchText='';
let dragAid=null;
let dayScroll={{}};
let newTourDay=null;
let scrollToNewDay=null;

function esc(v){{return String(v??'').replace(/[&<>"']/g,m=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m]))}}
function rad(x){{return x*Math.PI/180}}
function dist(a,b){{if(!Number.isFinite(a?.lat)||!Number.isFinite(a?.lon)||!Number.isFinite(b?.lat)||!Number.isFinite(b?.lon))return NaN;const R=6371,dlat=rad(b.lat-a.lat),dlon=rad(b.lon-a.lon),q=Math.sin(dlat/2)**2+Math.cos(rad(a.lat))*Math.cos(rad(b.lat))*Math.sin(dlon/2)**2;return 2*R*Math.asin(Math.sqrt(q))}}
function fmt(n){{return Number.isFinite(n)?n.toFixed(1):'–'}}
function routeId(day,source,tour){{return day+'::'+source+'::'+String(tour)}}
function activeSources(){{return SOURCES.filter(s=>document.getElementById('src_'+s)?.checked)}}
function getRoute(id){{return routes.find(r=>r.id===id)}}
function getAssignment(aid){{return assignments.find(a=>a.aid===aid)}}
function daySort(a,b){{const dd=DAY_ORDER.indexOf(a.day)-DAY_ORDER.indexOf(b.day);if(dd)return dd;const tt=String(a.tour).localeCompare(String(b.tour),'de',{{numeric:true,sensitivity:'base'}});return tt||a.source.localeCompare(b.source)}}

function initControls(){{
  document.getElementById('sourceChips').innerHTML=SOURCES.map(s=>`<label class="chip"><input type="checkbox" id="src_${{s}}" checked> ${{esc(s.replace('HUPA_','HUPA '))}}</label>`).join('');
  SOURCES.forEach(s=>document.getElementById('src_'+s).addEventListener('change',renderAll));
  document.getElementById('search').addEventListener('input',e=>{{searchText=e.target.value.trim().toLowerCase();renderAll()}});
  document.getElementById('resetBtn').addEventListener('click',()=>{{if(confirm('Alle manuellen Änderungen zurücksetzen?'))buildOriginalPlan()}});
  document.getElementById('csvBtn').addEventListener('click',downloadCSV);
  document.getElementById('newTourSource').innerHTML=SOURCES.map(s=>`<option value="${{s}}">${{esc(s.replace('HUPA_','HUPA '))}}</option>`).join('');
  document.getElementById('cancelTourBtn').addEventListener('click',closeNewTour);
  document.getElementById('createTourBtn').addEventListener('click',createNewTour);
  document.getElementById('tourModal').addEventListener('click',e=>{{if(e.target.id==='tourModal')closeNewTour()}});
  document.getElementById('newTourNo').addEventListener('keydown',e=>{{if(e.key==='Enter')createNewTour();if(e.key==='Escape')closeNewTour()}});
}}

function buildOriginalPlan(){{
  assignments=[]; routes=[]; unplanned=[];
  CUSTOMERS.forEach(c=>{{
    DAY_ORDER.forEach(day=>{{
      const tour=c[day];
      if(tour==null||tour==='')return;
      const a={{...c,aid:c.uid+'::'+day,originalDay:day,originalTour:String(tour),currentDay:day,currentTour:String(tour),originalRouteId:routeId(day,c.Quelle,tour)}};
      assignments.push(a);
      let r=routes.find(x=>x.id===a.originalRouteId);
      if(!r){{r={{id:a.originalRouteId,day,source:c.Quelle,tour:String(tour),items:[],originalCount:0}};routes.push(r)}}
      r.items.push(a); r.originalCount++;
    }})
  }});
  routes.forEach(r=>r.items.sort((a,b)=>(a.source_row??0)-(b.source_row??0)));
  routes.sort(daySort);
  renderAll();
}}

function routeChanged(r){{return r.items.length!==r.originalCount||r.items.some(a=>a.originalRouteId!==r.id)}}
function dayAssignments(day){{return routes.filter(r=>r.day===day).reduce((n,r)=>n+r.items.length,0)}}
function matches(a,r){{if(!searchText)return true;return [a.SAP,a.CSB,a.Name,a.Ort,a.Plz,a.Strasse,r?.tour,r?.source,DAY_LABELS[r?.day]].some(v=>String(v??'').toLowerCase().includes(searchText))}}

function duplicateOnTargetDay(a,targetRoute){{
  if(!targetRoute)return false;
  return routes.some(r=>r.day===targetRoute.day&&r.items.some(x=>x.aid!==a.aid&&String(x.SAP)===String(a.SAP)));
}}

function fitFor(a,r){{
  if(!r||!r.items)return {{key:'unknown',label:'keine Bewertung',nearest:NaN,nearName:'',centroid:NaN,dupe:false}};
  const others=r.items.filter(x=>x.aid!==a.aid&&Number.isFinite(x.lat)&&Number.isFinite(x.lon));
  const dupe=duplicateOnTargetDay(a,r);
  if(!Number.isFinite(a.lat)||!Number.isFinite(a.lon)||others.length===0)return {{key:'unknown',label:'nicht bewertbar',nearest:NaN,nearName:'',centroid:NaN,dupe}};
  let nearest=Infinity,near=null;
  others.forEach(x=>{{const d=dist(a,x);if(d<nearest){{nearest=d;near=x}}}});
  const clat=others.reduce((s,x)=>s+x.lat,0)/others.length,clon=others.reduce((s,x)=>s+x.lon,0)/others.length;
  const centroid=dist(a,{{lat:clat,lon:clon}});
  let key,label;
  if(nearest<=12&&centroid<=35){{key='good';label='passt sehr gut'}}
  else if(nearest<=25&&centroid<=55){{key='ok';label='passt gut'}}
  else if(nearest<=45&&centroid<=80){{key='warn';label='grenzwertig'}}
  else {{key='bad';label='eher nicht'}}
  if(dupe&&key==='good')key='ok';
  return {{key,label,nearest,nearName:near?`${{near.SAP}} · ${{near.Ort}}`:'',centroid,dupe}};
}}

function cardHtml(a,r){{
  const moved=a.originalRouteId!==r.id;
  const fit=moved?fitFor(a,r):null;
  const bad=fit?.key==='bad';
  const dupe=fit?.dupe;
  if(!matches(a,r))return '';
  // Sichtbar bleiben nur SAP, Straße, Name und Ort. Alle weiteren Daten
  // (CSB, PLZ, Quelle, Originaltag/-tour, Geo- und Bewertungsdaten) bleiben
  // vollständig im Objekt a bzw. in der Routenlogik erhalten.
  return `<div class="cust ${{moved?'moved':''}} ${{bad?'badmove':''}} ${{dupe?'dupe':''}}" draggable="true" data-aid="${{a.aid}}"><div class="grip">⋮</div><div class="cmain"><div class="cname">${{a.SAP}} · ${{esc(a.Strasse||'')}}</div><div class="csub"><strong>${{esc(a.Name)}}</strong> · ${{esc(a.Ort)}}</div></div><button class="remove" title="Ausplanen" data-remove="${{a.aid}}">×</button></div>`;
}}

function routeHtml(r){{
  if(!activeSources().includes(r.source))return '';
  const cards=r.items.map(a=>cardHtml(a,r)).join('');
  if(searchText&&!cards&&!r.manuallyCreated)return '';
  const d=r.items.length-r.originalCount;
  const meta=r.manuallyCreated?`${{r.items.length}} Kunden · neu angelegt`:`${{r.items.length}} Kunden · ursprünglich ${{r.originalCount}}`;
  return `<section class="route ${{routeChanged(r)?'changed':''}}" data-route="${{r.id}}"><div class="routehead"><div class="rleft"><div class="rtitle">Tour ${{esc(r.tour)}} <span class="source">${{esc(r.source.replace('HUPA_','HUPA '))}}</span></div><div class="rmeta">${{meta}}</div></div><div class="route-right">${{d!==0?`<span class="delta">${{d>0?'+':''}}${{d}}</span>`:''}}${{r.manuallyCreated?`<button class="route-delete" title="Neue Tour löschen" data-delete-route="${{r.id}}">×</button>`:''}}</div></div><div class="dropzone" data-route="${{r.id}}">${{cards||'<div class="empty">Kunden hierher ziehen</div>'}}</div></section>`;
}}

function renderWeek(){{
  const week=document.getElementById('week');
  week.innerHTML=DAY_ORDER.map(day=>{{
    const rs=routes.filter(r=>r.day===day).sort(daySort);
    const active=rs.filter(r=>activeSources().includes(r.source));
    const total=active.reduce((n,r)=>n+r.items.length,0);
    return `<div class="daycol"><div class="dayhead"><div class="dayhead-main"><div class="dayname">${{DAY_LABELS[day]}}</div><div class="daymeta">${{active.length}} Touren · ${{total}} Kunden</div></div><button class="addtour" type="button" data-add-tour="${{day}}">+ Tour</button></div><div class="daybody" data-day="${{day}}">${{rs.map(routeHtml).join('')||'<div class="empty">Keine Touren</div>'}}</div></div>`;
  }}).join('');
}}

function saveDayScroll(){{
  document.querySelectorAll('.daybody[data-day]').forEach(el=>dayScroll[el.dataset.day]=el.scrollTop);
}}
function restoreDayScroll(){{
  document.querySelectorAll('.daybody[data-day]').forEach(el=>{{
    if(scrollToNewDay===el.dataset.day)el.scrollTop=el.scrollHeight;
    else el.scrollTop=dayScroll[el.dataset.day]||0;
  }});
  scrollToNewDay=null;
}}
function openNewTour(day){{
  newTourDay=day;
  document.getElementById('newTourHint').textContent=`${{DAY_LABELS[day]}} · die Tour wird leer angelegt und erscheint sofort in dieser Tages-Spalte.`;
  document.getElementById('newTourNo').value='';
  document.getElementById('newTourError').textContent='';
  const firstActive=activeSources()[0]||SOURCES[0];
  document.getElementById('newTourSource').value=firstActive;
  const m=document.getElementById('tourModal');m.classList.add('show');m.setAttribute('aria-hidden','false');
  setTimeout(()=>document.getElementById('newTourNo').focus(),0);
}}
function closeNewTour(){{
  newTourDay=null;const m=document.getElementById('tourModal');m.classList.remove('show');m.setAttribute('aria-hidden','true');
}}
function createNewTour(){{
  const tour=String(document.getElementById('newTourNo').value||'').trim();
  const source=document.getElementById('newTourSource').value;
  const err=document.getElementById('newTourError');
  if(!newTourDay||!tour){{err.textContent='Bitte eine Tournummer oder einen Namen eingeben.';return}}
  const id=routeId(newTourDay,source,tour);
  if(getRoute(id)){{err.textContent='Diese Tour gibt es an diesem Tag in diesem Bereich bereits.';return}}
  routes.push({{id,day:newTourDay,source,tour,items:[],originalCount:0,manuallyCreated:true}});
  routes.sort(daySort);scrollToNewDay=newTourDay;closeNewTour();renderAll();
}}
function deleteNewRoute(id){{
  const r=getRoute(id);if(!r||!r.manuallyCreated)return;
  if(r.items.length&&!confirm(`Tour ${{r.tour}} löschen? Die ${{r.items.length}} enthaltenen Kunden werden nach AUSGEPLANT verschoben.`))return;
  if(r.items.length)unplanned.push(...r.items);
  routes=routes.filter(x=>x.id!==id);renderAll();
}}
function bindDayActions(){{
  document.querySelectorAll('[data-add-tour]').forEach(b=>b.addEventListener('click',()=>openNewTour(b.dataset.addTour)));
  document.querySelectorAll('[data-delete-route]').forEach(b=>b.addEventListener('click',e=>{{e.stopPropagation();deleteNewRoute(b.dataset.deleteRoute)}}));
}}

function renderUnplanned(){{
  const box=document.getElementById('unplannedDrop');
  box.innerHTML=unplanned.filter(a=>activeSources().includes(a.Quelle)&&(!searchText||[a.SAP,a.CSB,a.Name,a.Ort,a.Plz,a.Strasse].some(v=>String(v??'').toLowerCase().includes(searchText)))).map(a=>`<div class="cust moved" draggable="true" data-aid="${{a.aid}}"><div class="grip">⋮</div><div class="cmain"><div class="cname">${{a.SAP}} · ${{esc(a.Strasse||'')}}</div><div class="csub"><strong>${{esc(a.Name)}}</strong> · ${{esc(a.Ort)}}</div></div></div>`).join('')||'<div class="empty">Keine ausgeplanten Kunden</div>';
}}

function removeFromCurrent(aid){{
  const ui=unplanned.findIndex(a=>a.aid===aid);
  if(ui>=0)return {{a:unplanned.splice(ui,1)[0],from:null}};
  for(const r of routes){{const i=r.items.findIndex(a=>a.aid===aid);if(i>=0)return {{a:r.items.splice(i,1)[0],from:r}}}}
  return null;
}}

function moveAid(aid,toRouteId,beforeAid=null){{
  if(!aid)return;
  const found=removeFromCurrent(aid); if(!found)return;
  if(toRouteId==='UNPLANNED'){{unplanned.push(found.a);renderAll();showUnplanned(found.a);return}}
  const to=getRoute(toRouteId); if(!to){{if(found.from)found.from.items.push(found.a);else unplanned.push(found.a);renderAll();return}}
  let ix=beforeAid?to.items.findIndex(x=>x.aid===beforeAid):-1;
  if(ix<0)to.items.push(found.a);else to.items.splice(ix,0,found.a);
  found.a.currentDay=to.day;found.a.currentTour=to.tour;
  renderAll();
  if(found.a.originalRouteId!==to.id)showMove(found.a,to);
}}

function bindDnD(){{
  document.querySelectorAll('.cust').forEach(el=>{{
    el.addEventListener('dragstart',e=>{{dragAid=el.dataset.aid;el.style.opacity='.45';e.dataTransfer.effectAllowed='move'}});
    el.addEventListener('dragend',()=>{{el.style.opacity='';document.querySelectorAll('.dropzone,.unplanned-drop').forEach(z=>z.classList.remove('over'))}});
    el.addEventListener('dragover',e=>e.preventDefault());
    el.addEventListener('drop',e=>{{e.preventDefault();e.stopPropagation();const z=el.closest('.dropzone');if(z)moveAid(dragAid,z.dataset.route,el.dataset.aid)}});
  }});
  document.querySelectorAll('[data-remove]').forEach(b=>b.addEventListener('click',e=>{{e.stopPropagation();moveAid(b.dataset.remove,'UNPLANNED')}}));
  document.querySelectorAll('.dropzone,.unplanned-drop').forEach(z=>{{
    z.addEventListener('dragover',e=>{{e.preventDefault();z.classList.add('over')}});
    z.addEventListener('dragleave',()=>z.classList.remove('over'));
    z.addEventListener('drop',e=>{{if(e.target.closest('.cust'))return;e.preventDefault();z.classList.remove('over');moveAid(dragAid,z.dataset.route,null)}});
  }});
}}

function showMove(a,r){{
  const f=fitFor(a,r),d=document.getElementById('drawer');d.className='drawer show '+f.key;
  document.getElementById('drawerTitle').textContent=`${{DAY_LABELS[r.day]}} · Tour ${{r.tour}}: ${{f.label}}`;
  let text=`${{a.SAP}} · ${{a.Ort}}`;
  if(Number.isFinite(f.nearest))text+=` → nächster Kunde ${{f.nearName}}: ${{fmt(f.nearest)}} km; Abstand zum Tourzentrum ${{fmt(f.centroid)}} km.`;
  if(a.originalDay!==r.day)text+=` Liefertag geändert von ${{DAY_LABELS[a.originalDay]}} auf ${{DAY_LABELS[r.day]}}.`;
  if(f.dupe)text+=' Achtung: Dieser SAP-Kunde ist an dem neuen Tag bereits mit einer weiteren Lieferung vorhanden.';
  document.getElementById('drawerText').textContent=text;
  clearTimeout(window.__drawerTimer);window.__drawerTimer=setTimeout(()=>d.className='drawer',7000);
}}
function showUnplanned(a){{const d=document.getElementById('drawer');d.className='drawer show warn';document.getElementById('drawerTitle').textContent='Kunde ausgeplant';document.getElementById('drawerText').textContent=`${{a.SAP}} · ${{a.Name}} liegt jetzt im Bereich AUSGEPLANT.`;clearTimeout(window.__drawerTimer);window.__drawerTimer=setTimeout(()=>d.className='drawer',4500)}}

function renderMetrics(){{
  let moved=0,good=0;
  routes.forEach(r=>r.items.forEach(a=>{{if(a.originalRouteId!==r.id){{moved++;const f=fitFor(a,r);if(f.key==='good'||f.key==='ok')good++}}}}));
  document.getElementById('mDeliveries').textContent=assignments.length;
  document.getElementById('mRoutes').textContent=routes.filter(r=>activeSources().includes(r.source)).length;
  document.getElementById('mMoved').textContent=moved;
  document.getElementById('mGood').textContent=good;
  document.getElementById('mOut').textContent=unplanned.length;
}}

function renderChanges(){{
  const rows=[];
  routes.forEach(r=>r.items.forEach(a=>{{if(a.originalRouteId!==r.id){{const f=fitFor(a,r);const notes=[];if(a.originalDay!==r.day)notes.push('Tag geändert');if(f.dupe)notes.push('bereits Lieferung am Zieltag');rows.push([a.SAP,a.Name,`${{DAY_LABELS[a.originalDay]}} / ${{a.originalTour}}`,`${{DAY_LABELS[r.day]}} / ${{r.tour}}`,f.label,f.nearName,Number.isFinite(f.nearest)?fmt(f.nearest)+' km':'–',notes.join(', ')])}}}}));
  unplanned.forEach(a=>rows.push([a.SAP,a.Name,`${{DAY_LABELS[a.originalDay]}} / ${{a.originalTour}}`,'AUSGEPLANT','–','–','–','ausgeplant']));
  document.getElementById('changeBody').innerHTML=rows.length?rows.map(r=>`<tr>${{r.map(v=>`<td>${{esc(v)}}</td>`).join('')}}</tr>`).join(''):'<tr><td colspan="8" style="color:#888">Noch keine Änderungen.</td></tr>';
}}

function renderAll(){{saveDayScroll();renderWeek();renderUnplanned();renderMetrics();renderChanges();bindDnD();bindDayActions();restoreDayScroll()}}

function csvCell(v){{const s=String(v??'');const q=s.includes(';')||s.includes('\"')||s.includes(String.fromCharCode(10))||s.includes(String.fromCharCode(13));return q?'\"'+s.replace(/\"/g,'\"\"')+'\"':s}}
function downloadCSV(){{
  const rows=[['Status','Originaltag','Originaltour','Neuer Tag','Neue Tour','Bereich','Pos','CSB','SAP','Name','PLZ','Ort','Straße','Bewertung','Nächster Kunde','Distanz km','Tourzentrum km','Hinweis']];
  routes.forEach(r=>r.items.forEach((a,i)=>{{const moved=a.originalRouteId!==r.id,f=moved?fitFor(a,r):null,notes=[];if(moved&&a.originalDay!==r.day)notes.push('Tag geändert');if(f?.dupe)notes.push('bereits Lieferung am Zieltag');rows.push([moved?'verschoben':'unverändert',DAY_LABELS[a.originalDay],a.originalTour,DAY_LABELS[r.day],r.tour,a.Quelle,i+1,a.CSB??'',a.SAP??'',a.Name,a.Plz,a.Ort,a.Strasse||'',f?.label||'',f?.nearName||'',Number.isFinite(f?.nearest)?f.nearest.toFixed(1):'',Number.isFinite(f?.centroid)?f.centroid.toFixed(1):'',notes.join(', ')])}}));
  unplanned.forEach(a=>rows.push(['ausgeplant',DAY_LABELS[a.originalDay],a.originalTour,'AUSGEPLANT','',a.Quelle,'',a.CSB??'',a.SAP??'',a.Name,a.Plz,a.Ort,a.Strasse||'','','','','','']));
  const blob=new Blob(['\ufeff'+rows.map(r=>r.map(csvCell).join(';')).join(String.fromCharCode(13,10))],{{type:'text/csv;charset=utf-8'}}),link=document.createElement('a');
  link.href=URL.createObjectURL(blob);link.download='Feiertags_Wochenplanung.csv';link.click();URL.revokeObjectURL(link.href);
}}

initControls();buildOriginalPlan();
</script>
</body></html>'''


def main():
    import streamlit as st

    st.set_page_config(page_title="Feiertags-Wochenplaner V5 – HTML Generator", page_icon="📅", layout="wide")
    st.markdown(
        """
        <style>
        .block-container{max-width:1150px;padding-top:1.5rem}
        [data-testid="stFileUploader"]{background:#fafafa;border-radius:12px;padding:8px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Feiertags-Wochenplaner V5 – HTML Generator")
    st.caption("Aktuelle Excel hochladen → Wochenmatrix erzeugen → kompakte Kundenkacheln mit SAP, Straße, Name und Ort → alle Detaildaten bleiben für Berechnung und Export erhalten.")

    upload = st.file_uploader("Aktuelle Quelldatei (.xlsx)", type=["xlsx"])
    if upload is None:
        st.info("Benötigte Blätter: DIREKT, MK, HUPA_NMS und HUPA_MALCHOW.")
        return

    raw = upload.getvalue()
    try:
        with st.spinner("Quelldatei wird gelesen und PLZ werden geografisch ergänzt …"):
            df = read_source(raw)
            geo = enrich_geo(df)
            payload = make_payload(geo)
            html = build_html(payload, upload.name)
    except Exception as exc:
        st.error(f"Fehler: {exc}")
        return

    deliveries = sum(int(geo[d].notna().sum()) for d in DAY_COLUMNS)
    missing_geo = int(geo[["lat", "lon"]].isna().any(axis=1).sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("Kundenstammsätze", f"{len(geo):,}".replace(",", "."))
    c2.metric("Wochen-Lieferungen", f"{deliveries:,}".replace(",", "."))
    c3.metric("ohne Geo-Koordinate", missing_geo)

    st.success("HTML wurde erzeugt. In den Kundenkacheln werden nur SAP, Straße, Name und Ort angezeigt; alle weiteren Daten bleiben intern erhalten.")
    st.download_button(
        "Feiertags_Wochenplaner.html herunterladen",
        data=html.encode("utf-8"),
        file_name="Feiertags_Wochenplaner_V5.html",
        mime="text/html",
        use_container_width=True,
    )

    st.markdown(
        """
        **In der HTML:**
        - komplette Woche Montag–Samstag gleichzeitig
        - bestehende Touren bleiben unverändert
        - Kundenkachel zeigt nur **SAP, Straße, Name und Ort**
        - CSB, PLZ, Quelle, Originaltag/-tour und Geo-/Bewertungsdaten bleiben intern erhalten
        - **jeder Tag hat seine eigene Scrollleiste**
        - pro Tag über **+ Tour** neue Touren anlegen (auch z. B. FT-01)
        - Kunden zwischen Touren **und zwischen Tagen** per Drag & Drop verschieben
        - sofortige Bewertung der geografischen Passung
        - Warnung, wenn derselbe SAP-Kunde am Zieltag bereits eine Lieferung hat
        - Kunden ausplanen und später wieder einsetzen
        - Änderungstabelle + CSV-Export
        """
    )


if __name__ == "__main__":
    main()

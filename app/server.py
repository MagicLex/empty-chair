"""empty-chair -- server-rendered concealment auditor (no SPA).

Paste a UK company number or name. The app looks the company up in the precomputed
universe scores (all 5.7M companies), renders its disclosure evidence on the left
and a rank rail on the right, and streams a plain-language investigator's note from
the model's own numbers. Base experience is fully server-rendered and works with JS
off (POST /audit returns a complete page); streaming the note is a progressive
enhancement.

SIGNAL, NOT VERDICT. The score is a RELATIVE rank of how much a company's public
disclosure is SHAPED like structures where a hidden owner was later revealed. It is
never proof anyone hid anything, never an accusation, and most flagged companies are
legitimate (holding companies, family firms, dormant shells).
"""
import html
import json
import math
import os
import sys
import re
import time
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: F401

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import anthropic  # noqa: E402
import ask as A  # noqa: E402
import explain as E  # noqa: E402
from auditor import MODEL_VERSION  # noqa: E402
from chair_features import CONCEALMENT_FLAGS  # noqa: E402

DATA = os.path.join(CODE_DIR, "data", "universe_scores.parquet")
U = {"df": None, "by_num": None, "top": None, "at": 0.0, "error": ""}
ENGINE = {"client": None, "note": ""}
_PROXY_MOUNT = re.compile(r"^/hopsworks-api/pythonapp/[^/]+/[^/]+")


def _load():
    df = pd.read_parquet(DATA)
    df["company_number"] = df["company_number"].astype(str)
    # keep the app pod lean: downcast the 5.7M-row frame and hold ONE indexed copy
    # (index = company_number, kept as a column too for name search).
    for c in ("score", "pct_rank"):
        df[c] = df[c].astype("float32")
    for c in df.columns:
        if c not in ("company_number", "company_name", "sic_code", "company_status") \
                and str(df[c].dtype).startswith("int"):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    df = df.set_index("company_number", drop=False)
    U["df"] = df
    U["by_num"] = df
    U["top"] = df.nlargest(80, "score")
    # a real-company pool for "random", biased to companies that actually filed
    # something (skip the vast dormant/absent tail so a random pick is interesting)
    U["pool"] = df.index[(df["n_flags"] >= 1).values].to_numpy()
    # score distribution for the landing histogram (20 sharp bins)
    counts, edges = np.histogram(df["score"].to_numpy(), bins=20, range=(0, 1))
    U["hist"] = (counts.tolist(), [round(float(e), 2) for e in edges])
    # population base rate of each tell, for the audit-page fingerprint
    U["rates"] = {k: float(df[k].mean()) for k in CONCEALMENT_FLAGS if k in df.columns}
    try:
        U["nests"] = pd.read_parquet(os.path.join(CODE_DIR, "data", "linkage.parquet"))
        _build_webs()
        print(f"loaded {len(U['nests'])} nests, {len(U['webs'])} webs", flush=True)
    except Exception as e:
        U["nests"] = None
        U["webs"], U["wheels"] = [], []
        print(f"no linkage graph: {e}", flush=True)
    U["at"] = time.time()
    print(f"loaded {len(df)} companies", flush=True)


@asynccontextmanager
async def lifespan(app):
    try:
        _load()
    except Exception as e:
        U["error"] = str(e)
        print(f"load error: {e}", flush=True)
    try:
        key = None
        try:
            import hopsworks
            hopsworks.login()
            key = hopsworks.get_secrets_api().get_secret("ANTHROPIC_API_KEY").value
        except Exception as e:
            ENGINE["note"] = f"no anthropic key ({e}); dossier disabled"
        if key:
            ENGINE["client"] = anthropic.Anthropic(api_key=key)
    except Exception as e:
        ENGINE["note"] = str(e)
    yield


app = FastAPI(lifespan=lifespan)
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=2048)

# streamed responses must dodge the gzip middleware (it batches chunks) and tell
# any proxy on the path not to buffer, or tokens arrive in one burst at the end
STREAM_HEADERS = {"Content-Encoding": "identity", "X-Accel-Buffering": "no",
                  "Cache-Control": "no-cache"}


def stream_response(gen):
    return StreamingResponse(gen, media_type="text/plain; charset=utf-8",
                             headers=STREAM_HEADERS)

CSS = """
*{box-sizing:border-box}
:root{--paper:#efe9dc;--ink:#191712;--faint:#78716031;--dim:#6a6355;--red:#a8291c;
 --mono:"SFMono-Regular",Menlo,Consolas,monospace}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
 font:15px/1.5 Georgia,"Times New Roman",serif;
 background-image:repeating-linear-gradient(var(--paper),var(--paper) 27px,#0000 27px,#0000 28px)}
a{color:inherit;text-decoration:none;border-bottom:1px solid var(--ink)}
a:hover{background:var(--ink);color:var(--paper)}
.wrap{max-width:940px;margin:0 auto;padding:0 20px 80px}
.mast{border-bottom:3px double var(--ink);padding:22px 0 8px;margin-bottom:2px}
.mast h1{margin:0;font:800 2.1rem/1 var(--mono);letter-spacing:.16em;text-transform:uppercase}
.mast .reg{font:600 .7rem/1 var(--mono);letter-spacing:.34em;text-transform:uppercase;color:var(--dim);margin-top:8px}
.rule{display:flex;justify-content:space-between;font:600 .68rem/1 var(--mono);letter-spacing:.2em;
 text-transform:uppercase;color:var(--dim);border-bottom:1px solid var(--ink);padding:6px 0;margin-bottom:22px}
.note{font-size:.9rem;margin:0 0 22px;padding-left:14px;border-left:2px solid var(--red)}
.note b{font-family:var(--mono);font-size:.82rem;letter-spacing:.04em}
form.q{display:flex;gap:0;border:2px solid var(--ink);margin:0 0 8px}
input[type=text]{flex:1;min-width:180px;background:#0000;color:var(--ink);border:0;
 padding:13px 14px;font:1rem var(--mono)}
input[type=text]:focus{outline:0;background:#fff6}
button,.btn{background:var(--ink);color:var(--paper);border:2px solid var(--ink);cursor:pointer;
 font:700 .74rem/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;padding:0 18px}
button:hover,.btn:hover{background:var(--red);border-color:var(--red);color:#fff}
.acts{display:flex;gap:8px;margin:0 0 6px;flex-wrap:wrap}
.acts .btn{padding:9px 14px;display:inline-block}
.hint{color:var(--dim);font:.72rem/1.5 var(--mono);letter-spacing:.03em;margin:6px 0 0}
h2.sec{font:700 .78rem/1 var(--mono);letter-spacing:.22em;text-transform:uppercase;
 border-bottom:1px solid var(--ink);padding-bottom:6px;margin:34px 0 0}
/* score distribution: ramp-colored bars on paper */
.hist{display:flex;align-items:flex-end;gap:2px;height:70px;margin:14px 0 4px;border-bottom:1px solid var(--ink)}
.hist i{flex:1;min-height:1px}
.histx{display:flex;justify-content:space-between;font:.62rem var(--mono);color:var(--dim);letter-spacing:.1em}
/* the same distribution in the audit rail, with the company pinned on it */
.phist{display:flex;align-items:flex-end;gap:1px;height:52px;margin-top:8px;position:relative;
 border-bottom:1px solid var(--ink)}
.phist i{flex:1;min-height:1px}
.phist .pin{position:absolute;top:-5px;bottom:-1px;width:2.5px;background:var(--red);
 box-shadow:0 0 0 1.5px #efe9dc}
.pinlbl{color:var(--red);font-weight:700}
/* fingerprint: every tell with its population base rate */
.fp{display:flex;align-items:center;gap:10px;padding:5px 0;border-top:1px solid var(--faint);
 font:.74rem/1.4 var(--mono);color:var(--dim)}
.fp .fpl{flex:0 0 46%}
.fp .fpb{flex:1;height:9px;background:#19171212;position:relative}
.fp .fpb i{position:absolute;left:0;top:0;bottom:0;background:#9a8f7a}
.fp .fpn{flex:0 0 46px;text-align:right;font-variant-numeric:tabular-nums}
.fp.on{color:var(--ink)}
.fp.on .fpl{background:var(--ink);color:var(--paper);padding:3px 7px;font-weight:700}
.fp.on .fpl::before{content:"\\2588\\2588 ";color:var(--red);letter-spacing:-.1em}
.fp.on .fpb i{background:var(--red)}
.fp.on .fpn{color:var(--red);font-weight:700}
/* ledger */
table{width:100%;border-collapse:collapse;font-size:.9rem;margin-top:6px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--faint)}
th{font:600 .66rem/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--dim)}
tr:hover td{background:#fff5}
td.n,th.n{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td.no{font:.82rem var(--mono);color:var(--dim)}
.stage{display:grid;grid-template-columns:1fr 260px;gap:0;margin-top:22px;border:2px solid var(--ink)}
@media(max-width:780px){.stage{grid-template-columns:1fr}}
.filing{padding:18px 20px;border-right:2px solid var(--ink)}
@media(max-width:780px){.filing{border-right:0;border-bottom:2px solid var(--ink)}}
.filing h3{margin:0;font:1.3rem/1.15 Georgia,serif}
.filing .meta{font:.72rem/1.6 var(--mono);letter-spacing:.04em;color:var(--dim);margin:6px 0 16px;
 text-transform:uppercase}
.lbl{font:700 .66rem/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin:0 0 8px}
.lead{font:1.02rem/1.5 Georgia,serif;margin:0 0 16px;padding-bottom:14px;border-bottom:1px solid var(--faint)}
.clean{font:.82rem/1.5 var(--mono);color:var(--dim);padding:8px 0}
/* the stamp */
.rail{padding:18px 16px}
.stamp{border:3px solid var(--red);color:var(--red);padding:12px 8px;text-align:center;
 transform:rotate(-3deg);margin:4px 8px 16px}
.stamp .big{font:800 1.9rem/1 var(--mono);letter-spacing:.02em}
.stamp .cap{font:700 .58rem/1.3 var(--mono);letter-spacing:.2em;text-transform:uppercase;margin-top:5px}
.stamp.cold{border-color:var(--dim);color:var(--dim)}
.pctln{font:.66rem/1.5 var(--mono);color:var(--dim);text-align:center;letter-spacing:.08em;margin:-8px 0 16px}
.review{white-space:pre-wrap;font:.88rem/1.62 Georgia,serif;min-height:30px}
.review .lbl{margin-bottom:8px}
.web{margin:0 0 8px;border:2px solid var(--ink);padding:10px 12px 12px}
.web svg{display:block;width:100%;touch-action:none;cursor:grab}
.web figcaption{font:.72rem/1.4 var(--mono);margin-top:6px}
.web figcaption b{display:block;letter-spacing:.02em}
.web figcaption span{color:var(--dim);letter-spacing:.04em}
/* hydration states: ego-highlight dims everything outside the hovered node's web */
.web [data-d],.web [data-o],.web [data-ol],.web [data-e]{transition:opacity .12s}
.web.hl [data-d]:not(.hi),.web.hl [data-o]:not(.hi),.web.hl [data-ol]:not(.hi){opacity:.14}
.web.hl [data-e]:not(.hi){opacity:.05}
.pulse{animation:pl 1.5s ease-in-out infinite}
@keyframes pl{50%{stroke-opacity:.12}}
.tip{position:fixed;z-index:99;background:var(--ink);color:var(--paper);font:.75rem/1.55 var(--mono);
 padding:8px 11px;pointer-events:none;max-width:300px;box-shadow:3px 3px 0 #19171238}
.tip div{display:flex;gap:12px;justify-content:space-between}
.tip b{color:#fff;white-space:nowrap}
.tip span{color:#cfc8b8;text-align:right}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:0;
 border-top:1px solid var(--ink);margin-top:16px}
.grid .web{border:0;border-right:1px solid var(--faint);border-bottom:1px solid var(--ink);
 margin:0;padding:10px 12px 14px}
.leg{display:flex;gap:16px;flex-wrap:wrap;font:.66rem/1 var(--mono);color:var(--dim);
 letter-spacing:.08em;margin:10px 0;align-items:center}
.leg b{color:var(--ink)}
.leg .sw{display:inline-block;width:13px;height:10px;margin:0 1px;vertical-align:-1px}
.leg .sq{display:inline-block;width:9px;height:9px;border:2px solid var(--ink);
 vertical-align:-2px;background:var(--paper);padding:0}
.leg .sq.c{border-color:#2a6db5;border-radius:3px}
/* ask the register: floating panel, native <details> so it works without JS */
.askfloat{position:fixed;right:18px;bottom:18px;z-index:60;width:min(430px,calc(100vw - 36px))}
.askfloat summary{list-style:none;cursor:pointer;display:inline-block;float:right;
 background:var(--ink);color:var(--paper);font:700 .74rem/1 var(--mono);
 letter-spacing:.18em;text-transform:uppercase;padding:12px 18px;border:2px solid var(--ink);
 box-shadow:4px 4px 0 #19171240;user-select:none}
.askfloat summary::before{content:"\\25C8 ";color:var(--red)}
.askfloat summary::-webkit-details-marker{display:none}
.askfloat summary:hover{background:var(--red);border-color:var(--red)}
.askfloat[open] summary{background:var(--red);border-color:var(--red)}
.askp{clear:both;background:var(--paper);border:2px solid var(--ink);
 box-shadow:6px 6px 0 #19171240;padding:12px 14px;margin-top:8px;
 max-height:min(560px,72vh);display:flex;flex-direction:column;
 background-image:repeating-linear-gradient(var(--paper),var(--paper) 27px,#0000 27px,#0000 28px)}
.ctxchip{font:.66rem/1.5 var(--mono);color:var(--dim);letter-spacing:.04em;
 border-left:2px solid var(--red);padding:3px 8px;margin-bottom:8px}
.ctxchip b{color:var(--red);letter-spacing:.16em;text-transform:uppercase;margin-right:6px}
.alog{overflow-y:auto;flex:1;min-height:0}
.alog:empty{display:none}
.aq{font:700 .78rem/1.4 var(--mono);background:var(--ink);color:var(--paper);
 padding:6px 10px;margin:10px 0 6px;letter-spacing:.03em}
.aq::before{content:"Q // ";color:#cfc8b8}
.aa{font:.86rem/1.6 Georgia,serif;white-space:pre-wrap;overflow-wrap:break-word;
 padding:2px 0 10px;border-bottom:1px solid var(--faint)}
.alog{overscroll-behavior:contain}
form.askf{display:flex;gap:0;border:2px solid var(--ink);margin-top:10px;flex-shrink:0}
form.askf input[type=text]{padding:10px 11px;font-size:.85rem;min-width:0}
form.askf button{padding:0 14px}
form.askf .aclr{background:var(--paper);color:var(--dim);border:0;border-left:2px solid var(--ink);
 font-size:.9rem;padding:0 12px}
form.askf .aclr:hover{background:var(--red);color:#fff}
@media print{.askfloat{display:none}}
.foot{color:var(--dim);font:.68rem/1.6 var(--mono);letter-spacing:.06em;margin-top:44px;
 border-top:1px solid var(--ink);padding-top:12px;text-transform:uppercase}
"""

# Progressive enhancement over the SSR SVG: nearest-node hover with ego-highlight and
# tooltip, wheel/pinch zoom, drag-to-pan, drag-a-node with a spring sim seeded from the
# server layout (rest length = server distance, so the shape is preserved, not recomputed).
JS = r"""
(function(){
"use strict";
var TIP=document.createElement('div');TIP.className='tip';TIP.hidden=true;document.body.appendChild(TIP);
function tipShow(rows,x,y){TIP.textContent='';
 rows.forEach(function(r){var d=document.createElement('div'),b=document.createElement('b'),
  s=document.createElement('span');b.textContent=r[0];s.textContent=r[1];
  d.appendChild(b);d.appendChild(s);TIP.appendChild(d);});
 TIP.hidden=false;tipMove(x,y);}
function tipMove(x,y){var w=TIP.offsetWidth,h=TIP.offsetHeight,X=x+16,Y=y+16;
 if(X+w>innerWidth-8)X=x-w-16;if(Y+h>innerHeight-8)Y=y-h-16;
 TIP.style.left=X+'px';TIP.style.top=Y+'px';}
function tipHide(){TIP.hidden=true;}

document.querySelectorAll('.hist i,.phist i').forEach(function(b){
 b.addEventListener('pointermove',function(ev){
  var t=(b.getAttribute('data-c')||'').split(' · ');
  tipShow([[t[0]||'',t[1]||'']],ev.clientX,ev.clientY);});
 b.addEventListener('pointerleave',tipHide);});

// the graphs point the conversation: clicking an owner square retargets the chat
var askD=document.getElementById('ask');
if(askD)askD.addEventListener('toggle',function(){
 var log=askD.querySelector('.alog');
 if(askD.open&&log)log.scrollTop=log.scrollHeight;});

function setAskCtx(txt){
 var f=document.querySelector('form.askf'),chip=document.getElementById('askctx');
 if(!f)return;
 f.ctx.value=txt;
 if(chip){chip.hidden=false;chip.querySelector('span').textContent=txt;}
 var d=document.getElementById('ask');if(d)d.open=true;
 if(f.q&&!f.q.disabled)f.q.focus();
}

document.querySelectorAll('form.askf .aclr').forEach(function(b){
 b.addEventListener('click',function(){
  var f=b.closest('form');
  f.parentElement.querySelector('.alog').textContent='';
  f.hist.value='[]';f.q.value='';f.q.focus();});});

document.querySelectorAll('form.askf').forEach(function(f){
 f.addEventListener('submit',function(ev){
  if(!window.fetch||!window.ReadableStream)return;
  ev.preventDefault();
  var q=f.q.value.trim();if(!q||f.q.disabled)return;
  var log=f.parentElement.querySelector('.alog');
  var uq=document.createElement('div');uq.className='aq';uq.textContent=q;log.appendChild(uq);
  var ua=document.createElement('div');ua.className='aa';
  ua.textContent='consulting the register…';log.appendChild(ua);
  log.scrollTop=log.scrollHeight;
  f.q.value='';f.q.disabled=true;var first=true;
  function done(){f.q.disabled=false;f.q.focus();
   var h=[];try{h=JSON.parse(f.hist.value)||[];}catch(e){}
   h.push([q,ua.textContent]);f.hist.value=JSON.stringify(h.slice(-6));}
  var fd=new FormData();fd.append('q',q);fd.append('ctx',f.ctx.value);fd.append('hist',f.hist.value);
  fetch(f.getAttribute('action')+'/stream',{method:'POST',body:fd}).then(function(r){
   var rd=r.body.getReader(),dec=new TextDecoder();
   function pump(){return rd.read().then(function(x){
    if(x.done){done();return;}
    if(first){ua.textContent='';first=false;}
    var stick=log.scrollHeight-log.scrollTop-log.clientHeight<60;
    ua.textContent+=dec.decode(x.value,{stream:true});
    if(stick)log.scrollTop=log.scrollHeight;
    return pump();});}
   return pump();
  }).catch(function(e){ua.textContent+='\n[error: '+e+']';done();});
 });
});

document.querySelectorAll('figure.web').forEach(function(fig){
 var gd=fig.querySelector('script.gd'),svg=fig.querySelector('svg');
 if(!gd||!svg)return;
 var G=JSON.parse(gd.textContent),NO=G.o.length;
 // edges carry global node indices: owners 0..NO-1, dots NO+j
 var nodes=G.o.map(function(o){return{x:o.x,y:o.y,vx:0,vy:0,m:6,fix:false};}).concat(
           G.d.map(function(d){return{x:d.x,y:d.y,vx:0,vy:0,m:1,fix:false};}));
 var springs=G.e.map(function(e){var a=nodes[e[0]],b=nodes[e[1]];
  return{a:a,b:b,l:Math.hypot(a.x-b.x,a.y-b.y)};});
 var oEl=[],olEl=[],dEl=[],eEl=[],fEl=svg.querySelector('[data-f]');
 svg.querySelectorAll('[data-o]').forEach(function(el){oEl[+el.getAttribute('data-o')]=el;});
 svg.querySelectorAll('[data-ol]').forEach(function(el){olEl[+el.getAttribute('data-ol')]=el;});
 svg.querySelectorAll('[data-d]').forEach(function(el){dEl[+el.getAttribute('data-d')]=el;});
 svg.querySelectorAll('[data-e]').forEach(function(el){eEl[+el.getAttribute('data-e')]=el;});
 var adj=nodes.map(function(){return[];});
 G.e.forEach(function(e,i){adj[e[0]].push(i);adj[e[1]].push(i);});

 function redraw(){
  G.e.forEach(function(e,i){var a=nodes[e[0]],b=nodes[e[1]],el=eEl[i];if(!el)return;
   el.setAttribute('x1',a.x);el.setAttribute('y1',a.y);
   el.setAttribute('x2',b.x);el.setAttribute('y2',b.y);});
  G.d.forEach(function(d,j){var n=nodes[NO+j],el=dEl[j];
   if(el){el.setAttribute('cx',n.x);el.setAttribute('cy',n.y);}
   if(G.f===j&&fEl){fEl.setAttribute('cx',n.x);fEl.setAttribute('cy',n.y);}});
  G.o.forEach(function(o,k){var n=nodes[k],el=oEl[k];
   if(el){el.setAttribute('x',n.x-o.h);el.setAttribute('y',n.y-o.h);}
   var t=olEl[k];
   if(t){t.setAttribute('x',n.x);t.setAttribute('y',n.y+o.h+(+t.getAttribute('font-size'))+2);}});
 }

 var raf=null,dragN=null;
 function tick(){
  var i,j,a,b,dx,dy,d2,dl,f,fx,fy;
  for(i=0;i<springs.length;i++){var s=springs[i];
   dx=s.b.x-s.a.x;dy=s.b.y-s.a.y;dl=Math.hypot(dx,dy)||1e-6;
   f=.08*(dl-s.l)/dl;fx=f*dx;fy=f*dy;
   s.a.vx+=fx/s.a.m;s.a.vy+=fy/s.a.m;s.b.vx-=fx/s.b.m;s.b.vy-=fy/s.b.m;}
  for(i=0;i<NO;i++)for(j=i+1;j<NO;j++){a=nodes[i];b=nodes[j];
   dx=b.x-a.x;dy=b.y-a.y;d2=dx*dx+dy*dy+1;if(d2>25e4)continue;
   dl=Math.sqrt(d2);f=3200/d2;fx=f*dx/dl;fy=f*dy/dl;
   a.vx-=fx;a.vy-=fy;b.vx+=fx;b.vy+=fy;}
  var maxv=0;
  nodes.forEach(function(n){if(n.fix){n.vx=0;n.vy=0;return;}
   n.vx*=.8;n.vy*=.8;n.x+=n.vx;n.y+=n.vy;
   var v=n.vx*n.vx+n.vy*n.vy;if(v>maxv)maxv=v;});
  redraw();
  raf=(maxv>.02||dragN!==null)?requestAnimationFrame(tick):null;
 }
 function kick(){if(!raf)raf=requestAnimationFrame(tick);}

 var vb0=svg.getAttribute('viewBox').split(' ').map(Number),view=vb0.slice();
 function applyView(){svg.setAttribute('viewBox',view.join(' '));}
 function toWorld(cx,cy){var r=svg.getBoundingClientRect();
  return[view[0]+(cx-r.left)/r.width*view[2],view[1]+(cy-r.top)/r.height*view[3]];}
 function zoomAt(cx,cy,f){var nw=view[2]*f;
  if(nw<vb0[2]/14||nw>vb0[2]*3)return;
  var p=toWorld(cx,cy);
  view[0]=p[0]-(p[0]-view[0])*f;view[1]=p[1]-(p[1]-view[1])*f;
  view[2]*=f;view[3]*=f;applyView();}
 svg.addEventListener('wheel',function(ev){ev.preventDefault();
  zoomAt(ev.clientX,ev.clientY,Math.exp(ev.deltaY*.0012));},{passive:false});
 svg.addEventListener('dblclick',function(){view=vb0.slice();applyView();});

 function nearest(p){var best=null,bd=1e18;
  nodes.forEach(function(n,i){var dx=n.x-p[0],dy=n.y-p[1],d=dx*dx+dy*dy;
   if(d<bd){bd=d;best=i;}});
  var lim=Math.max(20,view[2]/38);
  return bd<lim*lim?best:null;}
 var lastHover=null;
 function highlight(i){
  if(i===lastHover)return;lastHover=i;
  var m={};m[i]=1;
  adj[i].forEach(function(ei){m['e'+ei]=1;m[G.e[ei][0]]=1;m[G.e[ei][1]]=1;});
  fig.classList.add('hl');
  oEl.forEach(function(el,k){if(el)el.classList.toggle('hi',!!m[k]);});
  olEl.forEach(function(el,k){if(el)el.classList.toggle('hi',!!m[k]);});
  dEl.forEach(function(el,j){if(el)el.classList.toggle('hi',!!m[NO+j]);});
  eEl.forEach(function(el,ei){if(el)el.classList.toggle('hi',!!m['e'+ei]);});}
 function clearHl(){if(lastHover===null)return;lastHover=null;
  fig.classList.remove('hl');
  svg.querySelectorAll('.hi').forEach(function(el){el.classList.remove('hi');});}
 function rowsFor(i){return (i<NO?G.o[i]:G.d[i-NO]).tip||[];}

 var ptrs={},panning=false,moved=0,last=null,pinch0=null;
 function nPtrs(){return Object.keys(ptrs).length;}
 svg.addEventListener('pointerdown',function(ev){
  ptrs[ev.pointerId]=[ev.clientX,ev.clientY];
  if(nPtrs()>1){pinch0=null;return;}
  var i=nearest(toWorld(ev.clientX,ev.clientY));moved=0;
  if(i!==null){dragN=i;nodes[i].fix=true;}
  else{panning=true;last=[ev.clientX,ev.clientY];}
  svg.setPointerCapture(ev.pointerId);});
 svg.addEventListener('pointermove',function(ev){
  if(ptrs[ev.pointerId])ptrs[ev.pointerId]=[ev.clientX,ev.clientY];
  if(nPtrs()===2){var v=Object.values(ptrs),
    d=Math.hypot(v[0][0]-v[1][0],v[0][1]-v[1][1]),
    mx=(v[0][0]+v[1][0])/2,my=(v[0][1]+v[1][1])/2;
   if(pinch0)zoomAt(mx,my,pinch0/d);
   pinch0=d;tipHide();return;}
  if(dragN!==null){var p=toWorld(ev.clientX,ev.clientY),n=nodes[dragN];
   n.x=p[0];n.y=p[1];moved++;kick();tipHide();clearHl();return;}
  if(panning){var r=svg.getBoundingClientRect();
   view[0]-=(ev.clientX-last[0])/r.width*view[2];
   view[1]-=(ev.clientY-last[1])/r.height*view[3];
   last=[ev.clientX,ev.clientY];moved++;applyView();return;}
  var i=nearest(toWorld(ev.clientX,ev.clientY));
  if(i===null){clearHl();tipHide();svg.style.cursor='grab';return;}
  svg.style.cursor='pointer';highlight(i);
  tipShow(rowsFor(i),ev.clientX,ev.clientY);});
 function release(ev){delete ptrs[ev.pointerId];pinch0=null;
  if(dragN!==null){nodes[dragN].fix=false;kick();}
  dragN=null;panning=false;}
 svg.addEventListener('pointerup',release);
 svg.addEventListener('pointercancel',release);
 svg.addEventListener('pointerleave',function(){clearHl();tipHide();});
 svg.addEventListener('click',function(ev){
  if(moved>3){ev.preventDefault();ev.stopPropagation();moved=0;return;}
  var i=nearest(toWorld(ev.clientX,ev.clientY));
  if(i===null)return;
  if(i>=NO){var el=dEl[i-NO],a=el&&el.parentElement,
    h=a&&(a.getAttribute('href')||(a.href&&a.href.baseVal));
   if(h){ev.preventDefault();ev.stopPropagation();window.open(h,'_blank');}}
  else{ev.preventDefault();ev.stopPropagation();
   var t=G.o[i].tip||[],parts=[];
   t.forEach(function(r){parts.push((r[0]?r[0]+': ':'')+r[1]);});
   setAskCtx('the graph node just clicked — '+parts.join(' · '));}},true);

 if(G.f!==null&&G.f!==undefined){
  // frame the focus company and its owners, with margin and enough of the web
  // around it for context; never tighter than 45% of the full extent
  var fi=NO+G.f,xs=[nodes[fi].x],ys=[nodes[fi].y];
  adj[fi].forEach(function(ei){var E=G.e[ei],n=nodes[E[0]===fi?E[1]:E[0]];
   xs.push(n.x);ys.push(n.y);});
  var x0=Math.min.apply(0,xs),x1=Math.max.apply(0,xs),
      y0=Math.min.apply(0,ys),y1=Math.max.apply(0,ys),
      cx=(x0+x1)/2,cy=(y0+y1)/2,
      z=Math.max((x1-x0)*2.6,(y1-y0)*2.6*vb0[2]/vb0[3],vb0[2]*.45,340);
  if(z<vb0[2]){var zh=z*vb0[3]/vb0[2];view=[cx-z/2,cy-zh/2,z,zh];applyView();}}
});
})();
"""

BAND = ('<p class=note><b>SIGNAL, NOT VERDICT.</b> This register ranks how much a company\'s '
        'public ownership <b>disclosure</b> is shaped like structures where a hidden owner was '
        'later revealed (ICIJ leaks, sanctions). It is never proof anyone hid anything. Most '
        'companies with this shape are legitimate: holding companies, family firms, dormant '
        'shells. The rank is relative, calibrated on a case-control sample, so read the '
        'position, not a probability.</p>')


# sequential score ramp (validated ordinal, light->dark on the paper surface) and the
# corporate-owner accent (validated categorical vs the ramp, CVD dE > 70)
RAMP = ("#c2907a", "#b56a4b", "#a84a2e", "#96291a", "#6e180d")
RAMP_CUTS = (0.5, 0.7, 0.85, 0.95)
CORP = "#2a6db5"


def ramp_of(score):
    for i, cut in enumerate(RAMP_CUTS):
        if score < cut:
            return RAMP[i]
    return RAMP[-1]


def esc(s):
    return html.escape(str(s if s is not None else ""))


def base(req):
    return (req.scope.get("root_path") or "").rstrip("/")


def severity(pct_rank):
    """(percentile 0-100, band label, is_hot). Percentile = share of companies that
    score LOWER, so high = more concealment-shaped. Honest: below the median gets a
    muted 'lower half', not an alarming 'top X%'."""
    pctl = int(round(float(pct_rank) * 100))
    if pctl >= 99:
        return pctl, "extreme", True
    if pctl >= 90:
        return pctl, "high", True
    if pctl >= 50:
        return pctl, "elevated", False
    return pctl, "lower half", False


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1:'st',2:'nd',3:'rd'}.get(n % 10,'th')}"


def fired(row):
    out = []
    for k, label in CONCEALMENT_FLAGS.items():
        v = row.get(k, 0)
        if v and int(v) > 0:
            out.append(label)
    return out


def page(bd, body, ask_ctx="", ask_pairs=None, ask_open=False):
    n = len(U["df"]) if U["df"] is not None else 0
    return (f"<!doctype html><html><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>EMPTY CHAIR &middot; register of concealment shape</title>"
            f"<style>{CSS}</style></head><body><div class=wrap>"
            f"<div class=mast><h1>Empty Chair</h1>"
            f"<div class=reg>Register of concealment shape &mdash; United Kingdom companies</div></div>"
            f"<div class=rule><span>Model v{MODEL_VERSION}</span><span>{n:,} companies on file</span>"
            f"<span>Signal, not verdict</span></div>"
            f"{BAND}"
            f"<form class=q method=post action='{bd}/audit'>"
            f"<input type=text name=q placeholder='COMPANY NUMBER (e.g. 12579660) OR NAME' autofocus>"
            f"<button>Audit</button></form>"
            f"<div class=acts>"
            f"<a class=btn href='{bd}/random'>Random company</a>"
            f"<a class=btn href='{bd}/random?hot=1'>Random high-shape</a>"
            f"<a class=btn href='{bd}/network'>Concealment nests</a></div>"
            f"<p class=hint>Every UK company is prescored. Paste a number for an exact match, a "
            f"name to search, or draw one at random.</p>{body}"
            f"<div class=foot>Empty Chair &middot; disclosure shape only, never intent &middot; "
            f"PU-labelled, scores are a lower bound &middot; built on Hopsworks</div>"
            f"</div>{ask_widget(bd, ctx=ask_ctx, pairs=ask_pairs, open_=ask_open)}"
            f"<script>{JS}</script></body></html>")


def _hist_bars(counts, edges):
    mx = max(counts) or 1
    return "".join(
        f"<i data-c='{c:,} companies &middot; score {edges[i]:.2f}&ndash;{edges[i + 1]:.2f}' "
        f"style='height:{max(1, round(100 * math.sqrt(c / mx)))}%;"
        f"background:{ramp_of((edges[i] + edges[i + 1]) / 2)}'></i>"
        for i, c in enumerate(counts))


def hist_html():
    counts, edges = U["hist"]
    return ("<h2 class=sec>Score distribution &mdash; 5.7M companies</h2>"
            f"<div class=hist>{_hist_bars(counts, edges)}</div>"
            "<div class=histx><span>0.0 &nbsp;clean</span><span>0.5 &nbsp;flag line</span>"
            "<span>1.0 &nbsp;concealment shape</span></div>")


def top_table(bd, top):
    rows = []
    for _, r in top.iterrows():
        pctl, band, _ = severity(r["pct_rank"])
        rows.append(
            f"<tr><td><a href='{bd}/audit?q={esc(r['company_number'])}'>{esc(r['company_name'])}</a></td>"
            f"<td class=no>{esc(r['company_number'])}</td>"
            f"<td class=n>{ordinal(pctl)}</td>"
            f"<td class=n>{int(r['n_flags'])}</td></tr>")
    return (hist_html()
            + "<h2 class=sec>Highest concealment shape on file</h2>"
            "<table><tr><th>Company</th><th>No.</th><th class=n>Rank</th><th class=n>Tells</th></tr>"
            + "".join(rows) + "</table>")


def resolve(q):
    q = (q or "").strip()
    if not q:
        return None
    key = q.upper().replace(" ", "")
    by = U["by_num"]
    if key in by.index:
        r = by.loc[key]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r.to_dict() | {"company_number": key}
    # name search
    df = U["df"]
    hit = df[df["company_name"].str.upper().str.contains(re.escape(q.upper()), na=False)]
    if len(hit):
        r = hit.sort_values("score", ascending=False).iloc[0]
        return r.to_dict()
    return None


def legend():
    sw = "".join(f"<i class=sw style='background:{c}'></i>" for c in RAMP)
    return ("<div class=leg>"
            f"<span>score {sw} 0&rarr;1</span>"
            "<span><b class=sq></b> person owner</span>"
            "<span><b class='sq c'></b> corporate owner</span>"
            "<span>&#9711; hollow = dissolved</span>"
            "<span style='color:var(--red)'>&#9472; shared company</span></div>")


def fingerprint(row):
    """Every tell as a row: fired ones dark, each with the population base rate as a
    bar. The rarity of a fired tell is the evidence; the reader sees it directly."""
    rates = U.get("rates") or {}
    if not rates:
        return ""
    mx = max(rates.values()) or 1
    rows = []
    for k, label in CONCEALMENT_FLAGS.items():
        rate = rates.get(k)
        if rate is None:
            continue
        on = bool(row.get(k)) and int(row.get(k, 0)) > 0
        rows.append(f"<div class='fp{' on' if on else ''}'>"
                    f"<span class=fpl>{esc(label)}</span>"
                    f"<span class=fpb><i style='width:{max(1.5, rate / mx * 100):.1f}%'></i></span>"
                    f"<span class=fpn>{rate * 100:.1f}%</span></div>")
    return ("<div class=lbl>Disclosure tells on file &mdash; and how common each is</div>"
            + "".join(rows)
            + "<p class=hint>Dark rows fired for this company. The bar is how much of the "
              "5.7M-company register shares that tell: the shorter the bar, the rarer "
              "the tell, the more it moves the score.</p>")


def pin_hist(row):
    counts, edges = U["hist"]
    score = float(row["score"])
    return (f"<div class=lbl style='margin-top:18px'>Where it sits</div>"
            f"<div class=phist>{_hist_bars(counts, edges)}"
            f"<b class=pin style='left:{min(99.0, score * 100):.1f}%'></b></div>"
            f"<div class=histx><span>0</span><span class=pinlbl>score {score:.2f}</span><span>1</span></div>")


def chair_fig(bd, row):
    """The disclosure structure every audited company gets, drawn from its own flags:
    the company, its registered office, and the ownership seat the public filings
    declare. No natural person disclosed = a dashed red square. The empty chair."""
    rates = U.get("rates") or {}
    score = float(row["score"])
    pctl, _, _ = severity(row["pct_rank"])

    def pct(k):
        return f"{rates.get(k, 0) * 100:.1f}% of UK companies share this trait"

    def on(k):
        v = row.get(k)
        return bool(v) and int(v) > 0

    CHAIR = {"stroke": "#a8291c", "dash": "7 5", "fill": "none", "sw": 2.6}
    FAINT = {"stroke": "#9a8f7a", "sw": 2}
    o, E = [], []

    def seat(x, y, name, kind, tip, h=15, style=None, edge=("C", 0)):
        E.append((edge[0] if edge[0] != "S" else len(o) - 1, len(o), edge[1]))
        o.append({"x": x, "y": y, "h": h, "name": name, "kind": kind, "nm": 0,
                  "label": True, "tip": tip, "style": style})

    if on("psc_super_secure") or on("psc_exempt"):
        k = "psc_super_secure" if on("psc_super_secure") else "psc_exempt"
        why = "super-secure protection" if on("psc_super_secure") else "a claimed PSC exemption"
        seat(300, 58, "IDENTITY WITHHELD", "sealed",
             [["sealed", f"the owner's identity is withheld from the public register via {why}"],
              ["rarity", pct(k)]], h=17, style={"stroke": "#191712", "fill": "#191712"}, edge=("C", 1))
    elif on("psc_absent") or on("psc_silence"):
        k = "psc_silence" if on("psc_silence") else "psc_absent"
        how = ("a filed no-PSC / not-identified / steps-not-completed statement"
               if on("psc_silence") else "no PSC record on file at all")
        seat(300, 58, "THE EMPTY CHAIR", "chair",
             [["empty", f"no person with significant control disclosed: {how}"],
              ["rarity", pct(k)]], h=17, style=CHAIR, edge=("C", 1))
    elif on("psc_corporate_only"):
        where = ("an entity registered outside the UK" if on("psc_foreign_corporate")
                 else "another company")
        seat(300, 104, "CORPORATE OWNER", "corporate",
             [["corporate", f"control is declared through {where}, not a person"],
              ["rarity", pct("psc_foreign_corporate" if on("psc_foreign_corporate")
                             else "psc_corporate_only")]])
        seat(300, 26, "WHO STANDS BEHIND IT?", "chair",
             [["empty", "the natural person behind the corporate owner is not on this register"]],
             h=13, style=CHAIR, edge=("S", 1))
    else:
        seat(300, 58, "NAMED PERSON ON FILE", "person",
             [["on file", "a natural person with significant control is publicly recorded"],
              ["", "the declared seat is occupied"]])
    if on("is_mill_address"):
        seat(105, 240, "FORMATION-MILL ADDRESS", "mill",
             [["mill", "registered at an address hosting large numbers of unrelated companies"],
              ["rarity", pct("is_mill_address")]], style={"stroke": "#a8291c", "sw": 2.4})
    else:
        seat(105, 240, "REGISTERED OFFICE", "office",
             [["office", "no formation-mill pattern at its registered address"]], style=FAINT)
    if on("accounts_dormant"):
        seat(495, 240, "DORMANT ACCOUNTS", "dormant",
             [["dormant", "files dormant / no-trading accounts"],
              ["rarity", pct("accounts_dormant")]], h=12,
             style={"stroke": "#9a8f7a", "dash": "3 3", "sw": 2})
    if on("is_holding_sic"):
        seat(495, 90, "HOLDING ACTIVITY", "holding",
             [["holding", "holding / management / trust activity code"],
              ["rarity", pct("is_holding_sic")]], h=12, style=FAINT)
    K = len(o)
    d = [{"x": 300, "y": 172, "num": row["company_number"], "name": row["company_name"],
          "score": round(score, 3), "active": _is_active(row.get("company_status")),
          "status": str(row.get("company_status") or ""), "bridge": False, "r": 15,
          "tip": [[f"{score:.2f}", row["company_name"]],
                  [ordinal(pctl), f"more concealment-shaped than {pctl}% of UK companies"],
                  [str(row.get("company_status") or "?"), f"SIC {row.get('sic_code') or '?'}"]]}]
    edges = [(K if a == "C" else a, b, c) for a, b, c in E]
    xs = [n["x"] for n in o + d]
    ys = [n["y"] for n in o + d]
    pad = 46
    vb = (min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
    w = {"owners": o, "dots": d, "edges": edges, "vb": vb}
    cap = (f"<b>{esc(row['company_name'])}</b>"
           "<span>the seats its public filings declare &middot; signal, not verdict</span>")
    return ("<h2 class=sec>Disclosure structure on file</h2>"
            "<p class=hint>What the register shows around this company. A dashed red square "
            "is a seat the filings leave empty; that absence is what the model scores. "
            "Drag nodes, hover for detail.</p>"
            + web_fig(bd, w, cap=cap))


def ego_fig(bd, row):
    """The company's ownership web, when it sits in a scored nest."""
    key = (U.get("comp_of") or {}).get(row["company_number"])
    if not key:
        return ""
    w = U["web_by_key"].get(key)
    if w is None:
        w = _web_layout(list(key), U["mem"], U["nests"])
    return ("<h2 class=sec>Ownership web on file</h2>"
            "<p class=hint>This company sits in a scored nest. Squares are its declared "
            "owners, dots the other companies they control. Drag, zoom, hover; the pulsing "
            "ring marks this company.</p>"
            + legend() + web_fig(bd, w, focus=row["company_number"]))


# ---- ask-the-register: deterministic tools the conversational layer may call ----

def _j(o):
    return json.dumps(o, default=str)


def _tool_lookup_company(args):
    row = resolve(str(args.get("q", "")))
    if row is None:
        return _j({"error": "no company matched"})
    pctl, band, _ = severity(row["pct_rank"])
    rates = U.get("rates") or {}
    tells = [{"tell": label, "population_rate_pct": round(rates.get(k, 0) * 100, 1)}
             for k, label in CONCEALMENT_FLAGS.items()
             if row.get(k) is not None and int(row.get(k, 0) or 0) > 0]
    return _j({"name": row["company_name"], "number": row["company_number"],
               "score_0to1": round(float(row["score"]), 3), "percentile": pctl,
               "band": band, "fired_tells": tells,
               "incorporated": row.get("incorporation_year"),
               "status": row.get("company_status"), "sic": row.get("sic_code"),
               "in_scored_nest": row["company_number"] in (U.get("comp_of") or {})})


def _tool_ownership_web(args):
    key = (U.get("comp_of") or {}).get(str(args.get("number", "")).upper().replace(" ", ""))
    if not key:
        return _j({"error": "this company does not sit in any scored nest"})
    nests, mem = U["nests"], U["mem"]
    owners = [{"owner": str(nests.iloc[i]["owner_name"]),
               "kind": str(nests.iloc[i]["owner_kind"]),
               "companies_in_nest": int(nests.iloc[i]["n_members"]),
               "mean_score": round(float(nests.iloc[i]["mean_score"]), 2)} for i in key]
    seen = {}
    for i in key:
        for x in mem[i]:
            e = seen.setdefault(x["number"], {"name": x["name"], "score": x["score"], "owners": 0})
            e["owners"] += 1
    by_score = sorted(seen.items(), key=lambda kv: -kv[1]["score"])
    return _j({"n_owners": len(owners), "owners": owners[:20],
               "n_companies": len(seen),
               "shared_companies": [{"number": n, **v} for n, v in by_score if v["owners"] > 1][:20],
               "highest_scoring_members": [{"number": n, "name": v["name"], "score": v["score"]}
                                           for n, v in by_score[:25]]})


def _tool_search_companies(args):
    sub = str(args.get("name_contains", "")).strip()
    if len(sub) < 3:
        return _j({"error": "give at least 3 characters"})
    df = U["df"]
    hit = df[df["company_name"].str.upper().str.contains(re.escape(sub.upper()), na=False)]
    hit = hit.nlargest(10, "score")
    return _j([{"name": r["company_name"], "number": r["company_number"],
                "score_0to1": round(float(r["score"]), 3),
                "percentile": severity(r["pct_rank"])[0]} for _, r in hit.iterrows()])


def _tool_top_ranked(args):
    n = min(int(args.get("limit") or 10), 15)
    return _j([{"name": r["company_name"], "number": r["company_number"],
                "score_0to1": round(float(r["score"]), 3),
                "fired_tell_count": int(r["n_flags"])}
               for _, r in U["top"].head(n).iterrows()])


def _tool_register_stats(args):
    rates = U.get("rates") or {}
    df = U["df"]
    return _j({"companies_scored": int(len(df)), "model_version": MODEL_VERSION,
               "tell_base_rates_pct": {CONCEALMENT_FLAGS[k]: round(v * 100, 1)
                                       for k, v in rates.items()},
               "share_scoring_above_0_5_pct": round(float((df["score"] > .5).mean()) * 100, 2),
               "scored_nests": int(len(U["nests"])) if U.get("nests") is not None else 0,
               "linked_webs_rendered": len(U.get("webs") or []),
               "note": "scores are relative concealment-shape ranks, signal not verdict"})


def _tool_owner_nests(args):
    name = str(args.get("owner_name", "")).strip()
    nests = U.get("nests")
    if nests is None or len(name) < 3:
        return _j({"error": "give at least 3 characters of an owner name"})
    hit = nests[nests["owner_name"].str.upper().str.contains(re.escape(name.upper()), na=False)]
    out = []
    for i in hit.index[:8]:
        r = nests.iloc[i]
        out.append({"owner": str(r["owner_name"]), "kind": str(r["owner_kind"]),
                    "companies_in_nest": int(r["n_members"]),
                    "mean_score": round(float(r["mean_score"]), 2),
                    "sample_members": [{"number": x["number"], "name": x["name"],
                                        "score": x["score"]} for x in U["mem"][i][:10]]})
    return _j(out or {"error": "no owner matched"})


ASK_TOOLS = {"lookup_company": _tool_lookup_company, "ownership_web": _tool_ownership_web,
             "search_companies": _tool_search_companies, "top_ranked": _tool_top_ranked,
             "register_stats": _tool_register_stats, "owner_nests": _tool_owner_nests}


def exec_tool(name, args):
    fn = ASK_TOOLS.get(name)
    return fn(args) if fn else _j({"error": f"unknown tool {name}"})


def ask_widget(bd, ctx="", pairs=None, open_=False):
    """Floating ask-the-register panel. A native <details> pinned bottom-right, so
    it opens and posts without JavaScript; the JS layer streams answers and lets the
    graphs update the context chip live."""
    rows = "".join(f"<div class=aq>{esc(q)}</div><div class=aa>{esc(a)}</div>"
                   for q, a in (pairs or []))
    return (f"<details class=askfloat id=ask{' open' if open_ else ''}>"
            f"<summary>Ask the register</summary><div class=askp>"
            f"<div class=ctxchip id=askctx {'hidden' if not ctx else ''}>"
            f"<b>on screen</b> <span>{esc(ctx)}</span></div>"
            f"<div class=alog>{rows}</div>"
            f"<form class=askf method=post action='{bd}/ask'>"
            f"<input type=hidden name=ctx value='{esc(ctx)}'>"
            f"<input type=hidden name=hist value='{esc(json.dumps(pairs or []))}'>"
            f"<input type=text name=q maxlength=300 autocomplete=off "
            f"placeholder='ASK ABOUT WHAT YOU SEE'>"
            f"<button>Ask</button>"
            f"<button type=button class=aclr title='clear the conversation'>&#10005;</button></form>"
            f"<div class=hint style='margin:6px 0 0'>Answers come from live register tools. "
            f"Click any owner square in a graph to point the conversation at it. "
            f"Signal, not verdict.</div></div></details>")


def evidence_card(row):
    flags = fired(row)
    pctl, band, _ = severity(row["pct_rank"])
    meta = (f"No. {esc(row['company_number'])} &nbsp;/&nbsp; inc. {esc(row.get('incorporation_year','?'))}"
            f" &nbsp;/&nbsp; {esc(row.get('company_status','?'))} &nbsp;/&nbsp; SIC {esc(row.get('sic_code','?'))}")
    lead = (f"Its public ownership disclosure is more concealment-shaped than <b>{pctl}%</b> "
            f"of UK companies. That is a structural resemblance, not a finding.")
    clean = "" if flags else "<div class=clean>No strong concealment tells on file for this company.</div>"
    return (f"<div class=filing><h3>{esc(row['company_name'])}</h3>"
            f"<div class=meta>{meta}</div>"
            f"<p class=lead>{lead}</p>{clean}"
            f"{fingerprint(row)}</div>")


def rail(bd, row, review_html):
    pctl, band, hot = severity(row["pct_rank"])
    return (f"<div class=rail><div class='stamp{'' if hot else ' cold'}'>"
            f"<div class=big>{ordinal(pctl)}</div>"
            f"<div class=cap>percentile &middot; {band}</div></div>"
            f"<div class=pctln>more concealment-shaped than {pctl}% of UK companies &middot; "
            f"{int(row['n_flags'])} tells</div>"
            f"{pin_hist(row)}"
            f"<div class=lbl style='margin-top:18px'>Investigator's note</div>"
            f"<div class=review id=review>{review_html}</div></div>")


def meta_of(row):
    return {k: row.get(k) for k in ("incorporation_year", "company_status", "sic_code")}


def _status_of(num):
    try:
        s = U["by_num"].at[num, "company_status"]
        return str(s) if s is not None else ""
    except Exception:
        return ""


def _is_active(status):
    return str(status).startswith("Active")


MAX_WEBS = 16  # network figures rendered; smaller webs fall back to the wheel grid


def _fr(n, edges_w, iters=250):
    """Fruchterman-Reingold on the owner graph. Deterministic golden-spiral init,
    so the layout is stable across restarts. n <= ~100, cheap."""
    idx = np.arange(n)
    pos = np.c_[np.sqrt(idx + .5) * np.cos(idx * 2.399963),
                np.sqrt(idx + .5) * np.sin(idx * 2.399963)]
    if n == 1 or not edges_w:
        return pos
    k, t = 1.0, 0.14
    for _ in range(iters):
        d = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((d ** 2).sum(-1)) + 1e-9
        disp = (d * (k * k / dist ** 2)[..., None]).sum(1)
        for i, j, w in edges_w:
            dv = pos[i] - pos[j]
            dl = math.hypot(*dv) + 1e-9
            f = dv * (dl / k) * min(w, 4) * .5
            disp[i] -= f
            disp[j] += f
        ln = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += disp / ln[:, None] * np.minimum(ln, t)[:, None]
        t *= .985
    return pos


def _web_layout(nest_idxs, mem, nests):
    """Resolve one connected web (owners bridged by co-owned companies) into final
    SVG geometry. Runs once at startup; requests only format strings."""
    K = len(nest_idxs)
    comp = {}
    for k, ni in enumerate(nest_idxs):
        for x in mem[ni]:
            c = comp.setdefault(x["number"], {"name": x["name"], "score": float(x["score"]), "ow": set()})
            c["ow"].add(k)
    excl = [[] for _ in range(K)]
    bridges = {}
    for num, c in comp.items():
        if len(c["ow"]) == 1:
            excl[next(iter(c["ow"]))].append((num, c))
        else:
            bridges.setdefault(tuple(sorted(c["ow"])), []).append((num, c))
    pair_w = Counter()
    for ow, lst in bridges.items():
        for a in range(len(ow)):
            for b in range(a + 1, len(ow)):
                pair_w[(ow[a], ow[b])] += len(lst)
    # bridge companies pack into a cols x rows phalanx between their owners; the
    # owners of a heavy pair must sit far enough apart to give the block room
    def _cols(n):
        return max(1, math.ceil(math.sqrt(1.8 * n)))

    extent = {}
    for ow, lst in bridges.items():
        if len(ow) == 2:
            key = (min(ow), max(ow))
            extent[key] = max(extent.get(key, 0.0), _cols(len(lst)) * 26.0)
    pos = _fr(K, [(i, j, w) for (i, j), w in pair_w.items()])
    nm_all = nests.iloc[nest_idxs]["n_members"].to_numpy()
    # ring must clear the owner square (7 + 2.4*sqrt(n_members) half-size)
    rr = np.array([max(24.0, (7 + 2.4 * math.sqrt(float(nm_all[k]))) * 1.7,
                       2.1 * min(len(excl[k]), 48)) for k in range(K)])
    if K > 1:
        # pack tight (median nn distance ~ ring scale), then relax collisions only.
        # a single worst-pair scale blows the canvas up with dead whitespace.
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)) + np.eye(K) * 1e9
        pos *= (float(rr.mean()) * 2.6) / max(float(np.median(d.min(1))), 1e-9)
        for _ in range(120):
            moved = False
            for i in range(K):
                for j in range(i + 1, K):
                    need = rr[i] + rr[j] + 30 + extent.get((i, j), 0.0)
                    dv = pos[i] - pos[j]
                    dl = math.hypot(*dv) + 1e-9
                    if dl < need:
                        pos[i] += dv * ((need - dl) / 2 / dl)
                        pos[j] -= dv * ((need - dl) / 2 / dl)
                        moved = True
            if not moved:
                break

    def dot(p, num, c, bridge):
        st = _status_of(num)
        return {"x": float(p[0]), "y": float(p[1]), "num": num, "name": c["name"],
                "score": c["score"], "active": _is_active(st), "status": st, "bridge": bridge}

    # edges carry GLOBAL node indices: owners are 0..K-1, dots are K+j
    dots, edges = [], []
    for k in range(K):
        ring = excl[k][:48]
        for t, (num, c) in enumerate(ring):
            a = 2 * math.pi * t / max(len(ring), 1) - math.pi / 2
            p = pos[k] + rr[k] * np.array([math.cos(a), math.sin(a)])
            edges.append((K + len(dots), k, 0))
            dots.append(dot(p, num, c, False))
    for ow, lst in bridges.items():
        ctr = pos[list(ow)].mean(0)
        if len(ow) == 2:
            dv = pos[ow[1]] - pos[ow[0]]
            dl = math.hypot(*dv) + 1e-9
            ax = dv / dl
            perp = np.array([-ax[1], ax[0]])
        else:
            ax, perp = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        cols = _cols(len(lst))
        rows = math.ceil(len(lst) / cols)
        for j, (num, c) in enumerate(lst):
            r_, c_ = divmod(j, cols)
            p = ctr + ax * ((c_ - (cols - 1) / 2) * 26.0) + perp * ((r_ - (rows - 1) / 2) * 26.0)
            for k in ow:
                edges.append((K + len(dots), k, 1))
            dots.append(dot(p, num, c, True))

    rows = nests.iloc[nest_idxs]
    nm = nm_all
    lbl_cut = sorted(nm, reverse=True)[min(9, K - 1)]
    owners = [{"x": float(pos[k][0]), "y": float(pos[k][1]),
               "h": 7 + 2.4 * math.sqrt(float(nm[k])),
               "name": str(rows.iloc[k]["owner_name"]), "kind": str(rows.iloc[k]["owner_kind"]),
               "nm": int(nm[k]), "label": bool(nm[k] >= lbl_cut)} for k in range(K)]

    xs = [d["x"] for d in dots] + [o["x"] for o in owners]
    ys = [d["y"] for d in dots] + [o["y"] for o in owners]
    pad = 44
    x0, y0 = min(xs) - pad, min(ys) - pad
    vb = (x0, y0, max(xs) + pad - x0, max(ys) + pad - y0)
    top = sorted(owners, key=lambda o: -o["nm"])[:5]
    return {"key": tuple(nest_idxs), "K": K, "n_comp": len(comp),
            "n_bridge": sum(len(v) for v in bridges.values()),
            "n_active": sum(1 for d in dots if d["active"]),
            "mean": float(np.mean([c["score"] for c in comp.values()])),
            "top": [o["name"] for o in top], "owners": owners, "dots": dots,
            "edges": edges, "vb": vb}


MAX_WHEELS = 36  # single-nest figures on /network


def _build_webs():
    """Union nests that share a company into connected webs. The big ones become
    full network figures; the hottest single nests render as small figures through
    the same code path. comp_of maps every nested company to its component so the
    audit page can lay out an ego web on demand."""
    nests = U["nests"]
    mem = [json.loads(m) for m in nests["members"]]
    U["mem"] = mem
    parent = list(range(len(mem)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners_of = {}
    for i, ms in enumerate(mem):
        for x in ms:
            owners_of.setdefault(x["number"], []).append(i)
    for lst in owners_of.values():
        for o in lst[1:]:
            parent[find(o)] = find(lst[0])
    comps = {}
    for i in range(len(mem)):
        comps.setdefault(find(i), []).append(i)
    heat = nests["mean_score"].to_numpy()
    multi = sorted((v for v in comps.values() if len(v) > 1),
                   key=lambda v: -sum(len(mem[i]) for i in v))
    singles = sorted((v[0] for v in comps.values() if len(v) == 1),
                     key=lambda i: -float(heat[i]))
    U["webs"] = [_web_layout(v, mem, nests) for v in multi[:MAX_WEBS]]
    U["wheels"] = [_web_layout([i], mem, nests) for i in singles[:MAX_WHEELS]]
    if len(multi) > MAX_WEBS:
        print(f"webs: rendering top {MAX_WEBS} of {len(multi)}, rest reachable via audit", flush=True)
    U["web_by_key"] = {w["key"]: w for w in U["webs"] + U["wheels"]}
    U["comp_of"] = {}
    for v in comps.values():
        t = tuple(v)
        for i in v:
            for x in mem[i]:
                U["comp_of"].setdefault(x["number"], t)


def _owner_tip(o):
    if o.get("tip"):
        return o["tip"]
    return [[str(o["nm"]), o["name"]], [o["kind"], f"owner of {o['nm']} high-shape companies"]]


def _dot_tip(d):
    if d.get("tip"):
        return d["tip"]
    tip = [[f"{d['score']:.2f}", d["name"]],
           ["active" if d["active"] else "dissolved", d["status"] or ""]]
    if d["bridge"]:
        tip.append(["shared", "held by more than one owner here"])
    return tip


def web_fig(bd, w, focus=None, small=False, cap=None):
    """One graph as SSR SVG (the no-JS baseline) plus a JSON payload the inline JS
    hydrates into pan/zoom, ego-highlight, tooltips and a drag force sim. Edges hold
    global node indices (owners 0..K-1, dots K+j) so any node can link to any node."""
    vb = w["vb"]
    K = len(w["owners"])

    def nxy(idx):
        return w["owners"][idx] if idx < K else w["dots"][idx - K]

    fs = max(2.5, vb[2] / 75)  # ~13px at typical full-width display; scales with zoom
    parts = []
    for i, (a, b, br) in enumerate(w["edges"]):
        na, nb = nxy(a), nxy(b)
        parts.append(f"<line data-e={i} x1={na['x']:.0f} y1={na['y']:.0f} x2={nb['x']:.0f} y2={nb['y']:.0f} "
                     f"stroke='{'#a8291c30' if br else '#1917121f'}' stroke-width={1.1 if br else 1} />")
    focus_i = None
    for j, d in enumerate(w["dots"]):
        rr = d.get("r") or 4 + 5 * d["score"]
        if d["active"]:
            fill, stroke, sw = ramp_of(d["score"]), ("#a8291c" if d["bridge"] else "none"), (1.6 if d["bridge"] else 0)
        else:
            fill, stroke, sw = "#efe9dc", ("#a8291c" if d["bridge"] else "#9a8f7a"), 1.5
        if focus is not None and d["num"] == focus:
            focus_i = j
            parts.append(f"<circle data-f=1 cx={d['x']:.0f} cy={d['y']:.0f} r={rr + 9:.1f} fill=none "
                         f"stroke='#191712' stroke-width=2.5 class=pulse />")
        parts.append(
            f"<a href='{bd}/audit?q={esc(d['num'])}' target=_blank>"
            f"<circle data-d={j} cx={d['x']:.0f} cy={d['y']:.0f} r={rr:.1f} fill='{fill}' "
            f"stroke='{stroke}' stroke-width={sw}>"
            f"<title>{esc(d['name'])} &middot; {d['score']} &middot; {esc(d['status'] or 'unknown')}"
            f"{' &middot; SHARED between owners' if d['bridge'] else ''}</title></circle></a>")
    for k, o in enumerate(w["owners"]):
        h = o["h"]
        st = o.get("style") or {}
        corp = o["kind"] == "corporate"
        rx = st.get("rx", 3 if corp else 0)
        stroke = st.get("stroke", CORP if corp else "#191712")
        dash = f"stroke-dasharray='{st['dash']}' " if st.get("dash") else ""
        parts.append(
            f"<rect data-o={k} x={o['x'] - h:.0f} y={o['y'] - h:.0f} width={2 * h:.0f} height={2 * h:.0f} "
            f"{f'rx={rx} ' if rx else ''}{dash}fill='{st.get('fill', '#efe9dc')}' "
            f"stroke='{stroke}' stroke-width={st.get('sw', 2.4)}>"
            f"<title>{esc(o['name'])} &middot; {esc(o['kind'])}</title></rect>")
        if o["label"] and not small:
            parts.append(f"<text data-ol={k} x={o['x']:.0f} y={o['y'] + h + fs + 2:.0f} text-anchor=middle "
                         f"font-family=monospace font-size={fs:.1f} fill='#6a6355' "
                         f"stroke='#efe9dc' stroke-width={fs / 4:.1f} paint-order=stroke>"
                         f"{esc(o['name'][:26])}</text>")
    payload = json.dumps({
        "o": [{"x": round(o["x"], 1), "y": round(o["y"], 1), "h": round(o["h"], 1),
               "tip": _owner_tip(o)} for o in w["owners"]],
        "d": [{"x": round(d["x"], 1), "y": round(d["y"], 1), "num": d["num"],
               "tip": _dot_tip(d)} for d in w["dots"]],
        "e": [list(e) for e in w["edges"]], "f": focus_i,
    }, separators=(",", ":"))
    if cap is None:
        bridged = f"{w['K']} owners bridged by {w['n_bridge']} shared companies" if w["K"] > 1 \
            else f"1 {w['owners'][0]['kind']} owner"
        cap = (f"<b>{' &middot; '.join(esc(n) for n in w['top'])}{' &hellip;' if w['K'] > 5 else ''}</b>"
               f"<span>{bridged} &middot; {w['n_comp']} companies &middot; "
               f"{w['n_active']} active &middot; mean {w['mean']:.2f}</span>")
    return (f"<figure class=web><svg viewBox='{vb[0]:.0f} {vb[1]:.0f} {vb[2]:.0f} {vb[3]:.0f}'>"
            f"<g class=vp>" + "".join(parts) + "</g></svg>"
            f"<script type='application/json' class=gd>{payload.replace('</', '<\\/')}</script>"
            f"<figcaption>{cap}</figcaption></figure>")


@app.get("/network", response_class=HTMLResponse)
async def network(req: Request):
    bd = base(req)
    nests = U.get("nests")
    if nests is None or not len(nests):
        return HTMLResponse(page(bd, "<p class=note>The linkage graph is not built yet. "
                                     "Run <b>build-linkage</b> to generate concealment nests.</p>"))
    body = ("<h2 class=sec>Concealment webs &mdash; owners bridged by shared companies</h2>"
            "<p class=note><b>NOT A LIST OF WRONGDOERS.</b> Controlling many companies with this "
            "shape is normal: property developers run one company per building, REITs and "
            "corporate-secretary agents appear on hundreds by trade. Nobody here is in the leaks; "
            "they only <b>match the shape</b> the leaks trained. Read it as where to look, never as "
            "who is guilty.</p>"
            "<p class=hint>A square is an owner, a dot is a high-shape company, a red line is a "
            "company shared between two owners: the bridge that stitches nests into one web. "
            "Drag any node, scroll to zoom, hover to isolate an owner's holdings. "
            "Click a dot to open that company in a new tab.</p>"
            + legend()
            + "".join(web_fig(bd, w) for w in U.get("webs") or [])
            + "<h2 class=sec>Isolated nests &mdash; hottest first</h2>"
            "<div class=grid>"
            + "".join(web_fig(bd, w, small=True) for w in U.get("wheels") or [])
            + "</div>")
    top_web = U["webs"][0] if U.get("webs") else None
    ctx = "the concealment webs page: shared-owner graphs over the top 1%"
    if top_web:
        ctx += f"; the largest web joins {top_web['K']} owners over {top_web['n_comp']} companies"
    return HTMLResponse(page(bd, body, ask_ctx=ctx))


@app.get("/random")
async def random_company(req: Request):
    from fastapi.responses import RedirectResponse
    bd = base(req)
    pool = U["top"].index.to_numpy() if req.query_params.get("hot") else U["pool"]
    if pool is None or not len(pool):
        return RedirectResponse(f"{bd}/", status_code=303)
    # no RNG seed dependence: pick by wall-clock nanoseconds modulo pool size
    idx = time.time_ns() % len(pool)
    return RedirectResponse(f"{bd}/audit?q={pool[idx]}", status_code=303)


@app.get("/", response_class=HTMLResponse)
async def home(req: Request):
    bd = base(req)
    if U["error"]:
        return HTMLResponse(page(bd, f"<div class=band>load error: {esc(U['error'])}</div>"))
    return HTMLResponse(page(bd, top_table(bd, U["top"]),
                             ask_ctx="the landing page: score distribution over 5.7M UK "
                                     "companies and the highest-ranked table"))


def _result(bd, row, review_html):
    pctl, _, _ = severity(row["pct_rank"])
    ctx = (f"the audit page for {row['company_name']} (company number "
           f"{row['company_number']}), ranked at the {ordinal(pctl)} percentile")
    return page(bd, f"<div class=stage>{evidence_card(row)}{rail(bd, row, review_html)}</div>"
                    f"{chair_fig(bd, row)}{ego_fig(bd, row)}", ask_ctx=ctx)


@app.post("/audit", response_class=HTMLResponse)
@app.get("/audit", response_class=HTMLResponse)
async def audit(req: Request, q: str = Form(default=None)):
    bd = base(req)
    if q is None:
        q = req.query_params.get("q")
    row = resolve(q)
    if row is None:
        return HTMLResponse(page(bd, f"<div class=band>No UK company matched "
                                     f"&ldquo;{esc(q)}&rdquo;.</div>" + top_table(bd, U["top"])))
    # no-JS fallback: compute the note synchronously
    review = "<span style='color:#7d8894'>dossier unavailable</span>"
    client = ENGINE["client"]
    if client:
        try:
            review = esc(E.explain(row["company_name"], float(row["pct_rank"]),
                                   [{"label": t} for t in fired(row)], meta_of(row), client))
        except Exception as e:
            review = f"<span style='color:#7d8894'>dossier error: {esc(e)}</span>"
    return HTMLResponse(_result(bd, row, review))


@app.get("/dossier")
async def dossier(req: Request):
    """Streamed investigator's note (progressive enhancement)."""
    row = resolve(req.query_params.get("q"))
    client = ENGINE["client"]
    if row is None or client is None:
        return stream_response(iter(["dossier unavailable"]))

    def gen():
        try:
            for delta in E.explain_stream(row["company_name"], float(row["pct_rank"]),
                                          [{"label": t} for t in fired(row)], meta_of(row), client):
                yield delta
        except Exception as e:
            yield f"\n[dossier error: {e}]"
    return stream_response(gen())


ASK_GAP = 3.0
_ask_last = {}


def _ask_pairs(hist):
    try:
        return [(str(h[0]), str(h[1])) for h in json.loads(hist or "[]")][-8:]
    except Exception:
        return []


def _throttled(req):
    ip = req.client.host if req.client else "?"
    now = time.time()
    if now - _ask_last.get(ip, 0.0) < ASK_GAP:
        return True
    if len(_ask_last) > 10000:
        _ask_last.clear()
    _ask_last[ip] = now
    return False


@app.post("/ask", response_class=HTMLResponse)
async def ask_page(req: Request, q: str = Form(...), ctx: str = Form(default=""),
                   hist: str = Form(default="[]")):
    """No-JS fallback: full round trip, whole page back with the transcript."""
    bd = base(req)
    pairs = _ask_pairs(hist)
    if _throttled(req):
        ans = "One question every few seconds, please."
    elif ENGINE["client"] is None:
        ans = "The conversational layer is offline (no model key configured)."
    else:
        try:
            ans = A.run_ask(q, pairs, ctx.strip() or None, ENGINE["client"], exec_tool)
        except Exception as e:
            ans = f"ask error: {e}"
    return HTMLResponse(page(bd, top_table(bd, U["top"]), ask_ctx=ctx,
                             ask_pairs=pairs + [(q, ans)], ask_open=True))


@app.post("/ask/stream")
async def ask_stream(req: Request, q: str = Form(...), ctx: str = Form(default=""),
                     hist: str = Form(default="[]")):
    """Progressive enhancement: streams tool status lines and answer tokens."""
    if _throttled(req):
        return stream_response(iter(["One question every few seconds, please."]))
    client = ENGINE["client"]
    if client is None:
        return stream_response(iter(["The conversational layer is offline."]))
    pairs = _ask_pairs(hist)

    def gen():
        try:
            for delta in A.run_ask_stream(q, pairs, ctx.strip() or None, client, exec_tool):
                yield delta
        except Exception as e:
            yield f"\n[ask error: {e}]"
    return stream_response(gen())


class StripForwardedPrefix:
    """Strip the Hopsworks proxy mount from the path (this cluster sets no
    APP_BASE_URL_PATH env and no X-Forwarded-Prefix header) so routes match, and
    record it as root_path for absolute links. Without it the app 404s forever."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            prefix = dict(scope.get("headers") or {}).get(b"x-forwarded-prefix", b"").decode().rstrip("/")
            if not prefix:
                m = _PROXY_MOUNT.match(scope["path"])
                prefix = m.group(0) if m else ""
            if prefix and scope["path"].startswith(prefix):
                scope = dict(scope)
                scope["path"] = scope["path"][len(prefix):] or "/"
                scope["root_path"] = prefix
        await self.inner(scope, receive, send)


application = StripForwardedPrefix(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)))

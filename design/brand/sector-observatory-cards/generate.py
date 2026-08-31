#!/usr/bin/env python3
"""Generate 4 claret/paper social cards from the Sector Observatory. Each card is a
standalone 1080x1350 HTML file; card 1 embeds the live heatmap canvas + data."""
import re, json, os

ROOT = "/Users/erikkins/CODE/stocker-app"
OUT = "/private/tmp/claude-501/-Users-erikkins-CODE-stocker-app/b87c584c-343d-4a11-aca7-a450196570be/scratchpad/cards"
os.makedirs(OUT, exist_ok=True)

html = open(f"{ROOT}/design/tools/sector-observatory.html").read()
DATA = re.search(r'<script id="DATA"[^>]*>(.*?)</script>', html, re.DOTALL).group(1)

BASE = """
<meta charset="utf-8">
<style>
  :root{--paper:#F5F1E8;--paper2:#EFE9DB;--card:#FBF8F0;--ink:#141210;--ink2:#3A342C;
    --claret:#7A2430;--warm:#8A8172;--rule:#E0D8C6;
    --serif:'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif;
    --sans:system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;
    --mono:ui-monospace,'SF Mono','IBM Plex Mono',Menlo,monospace;}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{background:#fff}
  .card{width:1080px;height:1350px;background:var(--paper);color:var(--ink);
    font-family:var(--sans);position:relative;overflow:hidden;
    padding:88px 84px;display:flex;flex-direction:column}
  .eyebrow{font-family:var(--mono);font-size:20px;letter-spacing:.24em;text-transform:uppercase;
    color:var(--claret);font-weight:600}
  h1{font-family:var(--serif);font-weight:600;letter-spacing:-.015em;line-height:1.0;
    text-wrap:balance;margin:.24em 0}
  .dek{font-family:var(--serif);font-style:italic;color:var(--ink2);line-height:1.32}
  .body{color:var(--ink2);line-height:1.5}
  .foot{position:absolute;left:84px;right:84px;bottom:64px;display:flex;justify-content:space-between;
    align-items:center;font-family:var(--mono);font-size:19px;color:var(--warm);
    border-top:1px solid var(--rule);padding-top:20px}
  .foot b{color:var(--claret);font-weight:600}
  .spacer{flex:1}
</style>
"""

FOOT = """<div class="foot"><span><b>RigaCap</b> · Sector Observatory</span><span>read-only research</span></div>"""

# ---------- Card 1: the heatmap hero ----------
card1 = f"""<!doctype html><html><head>{BASE}
<style>
  #heatwrap{{background:var(--card);border:1px solid var(--rule);border-radius:14px;
    padding:26px 22px 18px;margin-top:64px}}
  #heat{{display:block;width:100%}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px 30px;align-items:center;font-family:var(--mono);
    font-size:17px;color:var(--ink2);margin-top:22px}}
  .legend .grp{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}}
  .ramp{{display:inline-block;width:150px;height:14px;border-radius:2px;
    background:linear-gradient(90deg,#6B1F2A,#B4626A,#D8C7B4,#EDE6D6)}}
  .sw{{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px}}
</style></head>
<body><div class="card">
  <div class="eyebrow">RigaCap Research</div>
  <h1 style="font-size:82px;max-width:15ch">Leadership never sits still.</h1>
  <p class="dek" style="font-size:31px;max-width:52ch;margin-top:6px">Ten years of U.S. sector leadership &mdash;
    each month ranked and colored by the market regime that shaped it.</p>
  <div id="heatwrap">
    <canvas id="heat" height="300"></canvas>
    <div class="legend" id="legend"></div>
  </div>
  {FOOT}
</div>
<script id="DATA" type="application/json">{DATA}</script>
<script>
(function(){{
  const D=JSON.parse(document.getElementById('DATA').textContent);
  const SECT=D.sectors,M=D.months,S=D.summary,NREG=D.months.length;
  const REG={{strong_bull:{{c:'#B4622F',l:'Strong Bull'}},rotating_bull:{{c:'#7A2430',l:'Rotating Bull'}},
    weak_bull:{{c:'#C9A46A',l:'Weak Bull'}},range_bound:{{c:'#A79E8C',l:'Range-bound'}},
    recovery:{{c:'#77835A',l:'Recovery'}},weak_bear:{{c:'#5E6B78',l:'Weak Bear'}},panic_crash:{{c:'#2A2320',l:'Panic / Crash'}}}};
  const share=S.rotation_cadence.leader_share;
  const order=SECT.map((s,i)=>({{s,i,sh:share[s]||0}})).sort((a,b)=>b.sh-a.sh);
  function rankColor(r){{if(r==null)return '#EFE9DB';const t=(r-1)/(SECT.length-1);
    const a=[107,31,42],b=[237,230,214];const m=[0,1,2].map(k=>Math.round(a[k]+(b[k]-a[k])*Math.pow(t,0.85)));
    return 'rgb('+m[0]+','+m[1]+','+m[2]+')';}}
  const cv=document.getElementById('heat'),ctx=cv.getContext('2d');
  const LEFT=232,TOPBAND=24,GAP=14,ROWH=30,AXIS=34,ROWS=SECT.length;
  const H=TOPBAND+GAP+ROWS*ROWH+AXIS;
  const cssW=cv.clientWidth,dpr=2;
  cv.width=cssW*dpr;cv.height=H*dpr;cv.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0);
  const plotW=cssW-LEFT-8,cw=plotW/NREG;
  const WK=D.regimeWeekly,nwk=WK.length,wkw=plotW/nwk;
  for(let j=0;j<nwk;j++){{const rg=WK[j].r;ctx.fillStyle=(REG[rg]&&REG[rg].c)||'#CFC7B5';
    ctx.fillRect(LEFT+j*wkw,0,Math.ceil(wkw)+.8,TOPBAND);}}
  ctx.fillStyle='#8A8172';ctx.font='15px '+"'SF Mono',Menlo,monospace";ctx.textBaseline='middle';
  ctx.fillText('REGIME',LEFT-12-ctx.measureText('REGIME').width,TOPBAND/2);
  for(let ri=0;ri<ROWS;ri++){{const o=order[ri],y=TOPBAND+GAP+ri*ROWH;
    for(let j=0;j<NREG;j++){{ctx.fillStyle=rankColor(M[j].ranks[o.i]);ctx.fillRect(LEFT+j*cw,y,Math.ceil(cw)+.4,ROWH-3);}}
    ctx.fillStyle='#3A342C';ctx.font='18px system-ui,sans-serif';ctx.textAlign='right';
    ctx.fillText(o.s,LEFT-12,y+(ROWH-3)/2);ctx.textAlign='left';}}
  ctx.fillStyle='#8A8172';ctx.font='15px '+"'SF Mono',Menlo,monospace";ctx.textAlign='center';
  const yb=[];let ly=null;
  for(let j=0;j<NREG;j++){{const yr=M[j].d.slice(0,4);if(yr!==ly){{ly=yr;yb.push({{yr,x:LEFT+j*cw+cw/2}});}}}}
  for(let b=0;b<yb.length;b++){{const{{yr,x}}=yb[b];
    ctx.strokeStyle='#E0D8C6';ctx.beginPath();ctx.moveTo(x,TOPBAND+GAP-4);ctx.lineTo(x,TOPBAND+GAP+ROWS*ROWH);ctx.stroke();
    if(b+1<yb.length&&yb[b+1].x-x<44)continue;
    ctx.fillText(yr,x,TOPBAND+GAP+ROWS*ROWH+18);}}
  ctx.textAlign='left';
  const regsPresent=[...new Set(D.regimeWeekly.map(w=>w.r).filter(Boolean))]
    .sort((a,b)=>['strong_bull','rotating_bull','weak_bull','range_bound','recovery','weak_bear','panic_crash'].indexOf(a)
      -['strong_bull','rotating_bull','weak_bull','range_bound','recovery','weak_bear','panic_crash'].indexOf(b));
  document.getElementById('legend').innerHTML=
    '<span class="grp"><span>strongest</span><span class="ramp"></span><span>weakest</span></span>'+
    '<span class="grp">'+regsPresent.map(r=>'<span><span class="sw" style="background:'+REG[r].c+'"></span> '+REG[r].l+'</span>').join(' ')+'</span>';
}})();
</script>
</body></html>"""

# ---------- Card 2: persistence ----------
card2 = f"""<!doctype html><html><head>{BASE}
<style>
  .bars{{display:flex;align-items:flex-end;gap:40px;height:340px;margin:60px 0 8px;padding-top:20px}}
  .b{{flex:1;display:flex;flex-direction:column;align-items:center;gap:16px;height:100%;justify-content:flex-end}}
  .b .col{{width:100%;max-width:120px;border-radius:6px 6px 0 0}}
  .b .zero{{width:100%;max-width:120px;height:4px;background:var(--warm)}}
  .b .v{{font-family:var(--mono);font-size:30px;color:var(--ink);font-variant-numeric:tabular-nums}}
  .b .lb{{font-family:var(--mono);font-size:24px;color:var(--warm)}}
  .midl{{border-top:2px dashed var(--rule);margin-top:-172px;margin-bottom:172px}}
</style></head>
<body><div class="card">
  <div class="eyebrow">The half-life of a hot sector</div>
  <h1 style="font-size:74px">Momentum lasts a month,<br>not a quarter.</h1>
  <div class="bars">
    <div class="b"><div class="v">+0.56</div><div class="col" style="height:100%;background:var(--claret)"></div><div class="lb">1 month</div></div>
    <div class="b"><div class="v">&minus;0.03</div><div class="zero"></div><div class="lb">3 months</div></div>
    <div class="b"><div class="v">+0.09</div><div class="col" style="height:16%;background:var(--claret)"></div><div class="lb">6 months</div></div>
    <div class="b"><div class="v">&minus;0.05</div><div class="zero"></div><div class="lb">12 months</div></div>
  </div>
  <div class="midl"></div>
  <div class="body" style="font-size:30px;margin-top:24px">
    <p style="margin-bottom:26px">How strongly this month&rsquo;s leading sector predicts the next.</p>
    <p style="margin-bottom:26px">A firm <b style="color:var(--ink)">+0.56 at one month</b> collapses to a coin flip by three.</p>
    <p>Today&rsquo;s hot sector tells you almost nothing about next quarter&rsquo;s.</p>
  </div>
  {FOOT}
</div></body></html>"""

# ---------- Card 3: who leads when ----------
def rows_html():
    rl = json.loads(DATA)["summary"]["regime_leaders"]
    REGL = {"strong_bull":("Strong Bull","#B4622F"),"rotating_bull":("Rotating Bull","#7A2430"),
            "weak_bull":("Weak Bull","#C9A46A"),"range_bound":("Range-bound","#A79E8C"),
            "recovery":("Recovery","#77835A"),"weak_bear":("Weak Bear","#5E6B78"),"panic_crash":("Panic / Crash","#2A2320")}
    order = sorted(rl.items(), key=lambda kv: -kv[1]["n"])
    keep = [r for r in order if r[1]["n"] >= 2][:5]
    out = []
    for reg, o in keep:
        name, col = REGL.get(reg, (reg, "#999"))
        chips = "".join(f'<span class="chip">{t[0]}</span>' for t in o["top"][:3])
        out.append(f'<div class="rrow"><div class="rname"><span class="dot" style="background:{col}"></span>{name}</div><div class="chips">{chips}</div></div>')
    return "".join(out)

card3 = f"""<!doctype html><html><head>{BASE}
<style>
  .rrow{{display:flex;align-items:center;justify-content:space-between;gap:24px;
    padding:26px 0;border-bottom:1px solid var(--rule)}}
  .rname{{font-family:var(--serif);font-size:36px;color:var(--ink);display:flex;align-items:center;gap:18px;white-space:nowrap}}
  .dot{{width:22px;height:22px;border-radius:6px;flex:none}}
  .chips{{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}}
  .chip{{font-family:var(--mono);font-size:21px;padding:6px 16px;border-radius:99px;
    background:var(--paper2);color:var(--ink)}}
  .list{{margin:50px 0 0}}
</style></head>
<body><div class="card">
  <div class="eyebrow">The map</div>
  <h1 style="font-size:78px;max-width:15ch">Every market mood has its leaders.</h1>
  <p class="dek" style="font-size:29px;max-width:36ch;margin-top:4px">Who actually leads, sorted by how often each
    regime showed up over the last decade.</p>
  <div class="list">{rows_html()}</div>
  <div class="spacer"></div>
  {FOOT}
</div></body></html>"""

# ---------- Card 4: the verdict + CTA ----------
card4 = f"""<!doctype html><html><head>{BASE}
<style>
  .no{{font-family:var(--serif);font-weight:600;color:var(--claret);font-size:280px;line-height:.86;letter-spacing:-.03em}}
  .cta{{position:absolute;left:84px;right:84px;bottom:120px;font-family:var(--mono);font-size:24px;color:var(--ink)}}
  .cta b{{color:var(--claret)}}
</style></head>
<body><div class="card">
  <div class="eyebrow">The honest answer</div>
  <h1 style="font-size:70px;max-width:15ch;margin-top:.3em">Can you call the next hot sector?</h1>
  <div class="no">No.</div>
  <div class="body" style="font-size:30px;margin-top:16px">
    <p style="margin-bottom:26px">Past one month, leadership is unforecastable &mdash; we tested every &ldquo;early tell&rdquo; and none survived.</p>
    <p style="margin-bottom:26px">So we don&rsquo;t sell a crystal ball.</p>
    <p>We ride the one month that&rsquo;s real, and read the regime for context.</p>
  </div>
  <div class="cta">Read the full 10-year study &rarr; <b>link in bio</b></div>
</div></body></html>"""

for name, content in [("card1", card1), ("card2", card2), ("card3", card3), ("card4", card4)]:
    with open(f"{OUT}/{name}.html", "w") as f:
        f.write(content)
    print("wrote", name)
print("DONE")

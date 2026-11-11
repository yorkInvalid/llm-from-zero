"""课程 5 组件共用的画面样式：文字颜色跟随 VS Code / JupyterLab 主题，浅色、深色都可读。"""
from html import escape

CSS = """<style>
.tg{font:14px/1.6 system-ui,-apple-system,"PingFang SC",sans-serif;color:var(--vscode-editor-foreground,var(--jp-content-font-color1,#1f2328));overflow-wrap:anywhere}
.tg .muted,.tg small{color:var(--vscode-descriptionForeground,var(--jp-content-font-color2,#656d76))}
.tg h3{margin:4px 0 8px;font-size:17px}.tg p{margin:6px 0}
.tg code{font-family:ui-monospace,Menlo,monospace;font-size:13px;padding:0 3px;border-radius:4px;background:rgba(127,127,127,.14)}
.tg details{margin:6px 0}.tg summary{cursor:pointer}
.tg .head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.tg .q{font-size:20px;padding:10px 14px;margin:8px 0;border:1px solid rgba(127,127,127,.3);border-radius:8px;white-space:pre-wrap;word-break:break-all}
.tg .toks,.tg .chips{display:flex;flex-wrap:wrap;gap:4px;margin:10px 0}
.tg .tok{padding:2px 7px;border-radius:5px;border:1px solid;white-space:pre;font-size:15px}
.tg .chip{padding:3px 9px;border-radius:6px;border:1px solid rgba(127,127,127,.35);background:rgba(127,127,127,.08);white-space:pre-wrap}
.tg .chip.active{border-color:#2da44e;background:rgba(45,164,78,.16)}
.tg .in{animation:tg-in .3s ease-out both}
@keyframes tg-in{from{opacity:0;transform:translateY(8px) scale(.85)}to{opacity:1;transform:none}}
.tg .pop{display:inline-block;animation:tg-pop .45s ease-out both}
@keyframes tg-pop{0%{transform:scale(.6);opacity:0}60%{transform:scale(1.2);opacity:1}100%{transform:scale(1)}}
.tg .ruler{position:relative;height:76px;margin:12px 40px 6px}
.tg .ruler .line{position:absolute;left:0;right:0;top:37px;height:2px;background:rgba(127,127,127,.45)}
.tg .mark{position:absolute;transform:translateX(-50%);text-align:center;font-size:12px;white-space:nowrap}
.tg .mark.up{bottom:39px}.tg .mark.down{top:39px}
.tg .mark i{display:block;margin:0 auto;width:3px;height:14px;border-radius:2px;background:currentColor}
.tg .gap{position:absolute;top:35px;height:6px;border-radius:3px;opacity:.5}
.tg .chart{display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;align-items:center;margin-top:10px;font-size:13px}
.tg .track{position:relative;height:14px;border-radius:7px;background:rgba(127,127,127,.12)}
.tg .bar{position:absolute;left:0;top:0;bottom:0;border-radius:7px;opacity:.75}
.tg .tick{position:absolute;top:-4px;bottom:-4px;width:3px;margin-left:-1px;border-radius:2px;background:currentColor}
.tg .meter{height:12px;border-radius:6px;background:rgba(127,127,127,.15);overflow:hidden;margin:4px 0 10px}
.tg .meter .fill{height:100%;border-radius:6px}
.tg .big{font-size:24px;font-weight:650}.tg .metric{font-size:24px;font-weight:650}
.tg .hit{color:#2da44e}.tg .near{color:#d29922}.tg .miss{color:#e5534b}
.tg .win{border-left:3px solid #2da44e;padding:6px 12px;background:rgba(45,164,78,.08)}.tg .win.pop{display:block}
.tg .diff{outline:2px solid #e5534b;outline-offset:1px}
.tg table{border-collapse:collapse;font-size:13px;margin-top:8px}
.tg td,.tg th{padding:3px 10px;text-align:left;border-bottom:1px solid rgba(127,127,127,.22)}
@media(prefers-reduced-motion:reduce){.tg .in,.tg .pop{animation:none}}
</style>"""


def frame(body):
    return CSS + '<div class="tg">' + body + '</div>'


def chip(label, active=False):
    return '<span class="chip' + (' active' if active else '') + '">' + escape(str(label)) + '</span>'

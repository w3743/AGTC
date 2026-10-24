"""Native Isaac Sim editor for a paused representative scene; never steps training."""
from pathlib import Path
import os,sys,json,time,traceback
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
sys.argv.extend(['--portable-root','C:/pioneer/isaac-training/gui','--/app/settings/fabricDefaultStageFrameHistoryCount=3'])
from isaacsim import SimulationApp
app=None
try:
    app=SimulationApp({'headless':False,'hide_ui':False,'width':960,'height':540,
        'window_width':1500,'window_height':900,'multi_gpu':False,'active_gpu':0,
        'physics_gpu':0,'limit_cpu_threads':2,'anti_aliasing':2,
        'open_usd':str(ROOT/'current_training_config_static.usda')},
        experience='C:/pioneer/isaac-training/.venv/Lib/site-packages/isaacsim/apps/isaacsim.exp.full.kit')
    import omni.timeline,omni.usd,omni.ui as ui
    timeline=omni.timeline.get_timeline_interface()
    timeline.stop()
    from isaacsim.core.utils.viewports import set_camera_view
    import numpy as np
    for _ in range(8): app.update()
    set_camera_view(eye=np.array([3.4,0.2,1.4]),target=np.array([2.0,2.0,.15]))
    notice=ui.Window('Training scene - static preview',width=455,height=135)
    with notice.frame:
        with ui.VStack(spacing=6):
            ui.Label('12 m x 12 m | flat | plants=0 | rocks=0')
            ui.Label('Representative scene; timeline stopped.')
            ui.Label('Background training continues separately.')
            ui.Label('No live training state is attached.')
    (ROOT/'gui_ready.json').write_text(json.dumps({'pid':os.getpid(),'scene':str(ROOT/'current_training_config_static.usda'),
        'timeline_playing':timeline.is_playing(),'live_training_attached':False,'training_modified':False},indent=2),encoding='utf-8')
    while app.is_running():
        app.update()
        time.sleep(.1)
except BaseException:
    (ROOT/'gui_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    if app is not None: app.close()

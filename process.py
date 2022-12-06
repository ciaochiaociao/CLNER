from datetime import datetime, timedelta
import time
import psutil


def wait_for_process(pid_to_wait, show_dynamic_status=False):
    
    if pid_to_wait is not None:
        p = psutil.Process(pid_to_wait)

        print(f'Waiting for {p.pid} to finish ...')
        proc_info(p)
        if not show_dynamic_status:
            p.wait()
        else:
            while psutil.pid_exists(pid_to_wait):
                time.sleep(2)
                dynamic_status(p)
        print(f'\n >> {pid_to_wait} is finished at {time.ctime(time.time())}')


def proc_info(p):
    print(f'pid {p.pid}\'s command: {p.cmdline()}')
    
    def _proc_info(p):
        for i, c in enumerate(p.children()):
            print(f'{p.pid}\'s {i}th child >> {c.pid}\'s command: {c.cmdline()}')
            _proc_info(c)
    
    _proc_info(p)


def dynamic_status(p: psutil.Process):

    def t(p: psutil.Process):
        return timedelta(seconds=time.time()-p.create_time())
    print(f'{p.pid}\'s status: {p.status()} / time: {t(p)} |', end=' ')
    for i, c in enumerate(p.children()):
        print(f'{c.pid}\'s status: {c.status()} / time: {t(c)} |', end=' ')
    print(end='\r')
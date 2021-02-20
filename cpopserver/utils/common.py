import os
import time
import uuid
import logging
import tarfile
import threading
import traceback
import subprocess
import psutil
import requests

LOG = logging.getLogger(__name__)

DEFAULT_ENCODING = 'UTF-8'


class FuncThread(threading.Thread):
    """ Helper class to run a Python function in a background (daemon) thread. """

    def __init__(self, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.daemon = True
        self.func = func
        self.func_args = args
        self.func_kwargs = kwargs

    def run(self):
        try:
            self.func(*self.func_args, **self.func_kwargs)
        except Exception as e:
            LOG.warning('Thread run method %s failed: %s %s' %
                (self.func, e, traceback.format_exc()))

    def stop(self):
        LOG.warning('Not implemented: FuncThread.stop(..)')


class ShellCommandThread(FuncThread):
    """ Helper class to run a shell command in a background thread. """

    def __init__(self, cmd, log_result=False):
        self.cmd = cmd
        self.log_result = log_result
        FuncThread.__init__(self, self.run_cmd)

    def run_cmd(self):
        self.process = run(self.cmd, asynchronous=True)
        result = self.process.communicate()
        if self.log_result:
            LOG.info('Command "%s" produced output: %s' % (self.cmd, result))

    def stop(self):
        LOG.info('Terminating process: %s' % self.process.pid)
        kill_process_tree(self.process.pid)


def run(command, asynchronous=False, quiet=True):
    """ Run a shell command and return the output as a string """
    if not asynchronous:
        return to_str(subprocess.check_output(command, shell=True))
    kwargs = {'stdout': subprocess.DEVNULL} if quiet else {}
    process = subprocess.Popen(command, stderr=subprocess.STDOUT, shell=True, **kwargs)
    return process


def find_command(cmd):
    try:
        return run('which %s' % cmd).strip()
    except Exception:
        pass


def get_os_type():
    if is_mac_os():
        return 'osx'
    if is_alpine():
        return 'alpine'
    if is_linux():
        return 'linux'
    raise Exception('Unable to determine operating system')


def is_mac_os():
    return 'Darwin' in get_uname()


def is_linux():
    return 'Linux' in get_uname()


def is_alpine():
    try:
        if not os.path.exists('/etc/issue'):
            return False
        issue = to_str(run('cat /etc/issue'))
        return 'Alpine' in issue
    except subprocess.CalledProcessError:
        return False


def short_uid():
    return str(uuid.uuid4())[0:8]


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def download(url, target):
    response = requests.get(url, allow_redirects=True)
    with open(target, 'wb') as f:
        f.write(response.content)


def untar(path, target_dir):
    mode = 'r:gz' if path.endswith('gz') else 'r'
    with tarfile.open(path, mode) as tar:
        tar.extractall(path=target_dir)


def get_uname():
    try:
        return run('uname -a')
    except Exception:
        return ''


def kill_process_tree(parent_pid):
    parent_pid = getattr(parent_pid, 'pid', None) or parent_pid
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except Exception:
            pass
    parent.kill()


def sleep_forever():
    while True:
        time.sleep(1)


def retry(function, retries=3, sleep=1, sleep_before=0, **kwargs):
    raise_error = None
    if sleep_before > 0:
        time.sleep(sleep_before)
    for i in range(0, retries + 1):
        try:
            return function(**kwargs)
        except Exception as error:
            raise_error = error
            time.sleep(sleep)
    raise raise_error


def to_str(obj, encoding=DEFAULT_ENCODING, errors='strict'):
    """ If ``obj`` is an instance of ``bytes``, return
    ``obj.decode(encoding, errors)``, otherwise return ``obj`` """
    return obj.decode(encoding, errors) if isinstance(obj, bytes) else obj


def to_bytes(obj, encoding=DEFAULT_ENCODING, errors='strict'):
    """ If ``obj`` is an instance of ``str``, return
    ``obj.encode(encoding, errors)``, otherwise return ``obj`` """
    return obj.encode(encoding, errors) if isinstance(obj, str) else obj

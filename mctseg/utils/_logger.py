import json
import os
import subprocess
import datetime


class GlobalLogger(object):
    _instance = None
    _d = dict()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalLogger, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def update(self, tag, value, dtype=None):
        """
        Updates the internal state of the logger.

        Parameters
        ----------
        tag : str
            Tag, of the variable, which we log.
        value : object
            The value to be logged
        dtype :
            Container which is used to store the values under the tag

        Returns
        -------

        """
        if tag not in self._d:
            if dtype is None:
                self._d[tag] = (value, str(datetime.datetime.now()))
            else:
                self._d[tag] = dtype()
        else:
            if isinstance(self._d[tag], list):
                self._d[tag].append((value, str(datetime.datetime.now())))
            elif isinstance(self._d[tag], dict):
                self._d[tag].update((value, str(datetime.datetime.now())))
            else:
                self._d[tag] = value

    def json(self, indent=4):
        """
        Dumps the content of the logger into JSON-formatted string

        Parameters
        ----------
        indent : int
            Indentation level to be used

        Returns
        -------
        out : str
            JSON-formatted string, the content of the logger

        """
        return json.dumps(self._d, indent=indent)

    def __repr__(self):
        return self.json()

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.json())
            f.flush()


# Return the git revision as a string
def git_info():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')

        out = _minimal_ext_cmd(['git', 'rev-parse','--abbrev-ref' ,'HEAD'])
        git_branch = out.strip().decode('ascii')
    except OSError:
        return None

    return git_branch, git_revision


if __name__ == "__main__":
    logger = GlobalLogger()
    res = git_info()
    if res is not None:
        logger.update('git branch name', res[0])
        logger.update('git commit id', res[1])
    print(logger)


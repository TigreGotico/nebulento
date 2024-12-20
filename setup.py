import os
import os.path

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


def get_version():
    version_file = os.path.join(BASEDIR, 'nebulento', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if int(alpha):
        version += f"a{alpha}"
    return version


def get_description():
    with open(os.path.join(BASEDIR, "README.md"), "r") as f:
        long_description = f.read()
    return long_description


PLUGIN_ENTRY_POINT = 'ovos-nebulento-pipeline-plugin=nebulento.opm:NebulentoPipeline'


setup(
    name='nebulento',
    version=get_version(),
    packages=['nebulento'],
    url='https://github.com/OpenJarbas/nebulento',
    license='apache-2.0',
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    install_requires=required('requirements.txt'),
    description='dead simple fuzzy matching intent parser',
    long_description=get_description(),
    long_description_content_type="text/markdown",
    entry_points={'opm.pipeline': PLUGIN_ENTRY_POINT}
)

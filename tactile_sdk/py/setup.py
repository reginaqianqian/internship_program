from setuptools import setup, find_packages

with open('../version') as f:
    version = f.read().strip()
with open('../version_suffix') as f:
    version += f.read().strip()

# make package platform specific
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False
except ImportError:
    bdist_wheel = None

setup(
    name='tactile_sdk',
    version=version,
    description='',
    author='',
    author_email='qiyue@hesaitech.com',
    packages=find_packages() + ['app'],
    package_data={
        'tactile_sdk': ['sharpa/tactile/*.so']
    },
    keywords=[''],
    setup_requires=["pybind11"],
    install_requires=[
        'pybind11',
        'numpy',
        "ping3",
        "zmq"
    ],
    entry_points={
        'console_scripts': [
            'tactile-pub=app.tactile_pub:main',
            'tactile-pub-zmq=app.tactile_pub_zmq:main',
            'board-cfg=app.board_cfg:main',
            'board-update=app.board_update:main',
        ],
    },
    # install_requires=_parse_requirements('requirements.txt'),
    python_requires='>=3.10.0, <3.13.0',
    cmdclass={'bdist_wheel': bdist_wheel},
)

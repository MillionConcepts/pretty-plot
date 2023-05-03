from setuptools import setup, find_packages

setup(
      name="pretty_plot",
      version="0.2.0",
      url="https://github.com/millionconcepts.pretty-plot.git",
      packages=find_packages(),
      package_data={
            "pretty_plot": ['static/*.*', 'data/*.*']
      }
)

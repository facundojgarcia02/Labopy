from setuptools import setup, find_packages


setup(
    name='Labopy',
    version='0.1.1',
    license='MIT',
    author="Facundo Joaquin Garcia",
    author_email='facundojgarcia02@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/facundojgarcia02/Labopy',
    keywords='Labopy',
    install_requires=[
          'numpy','sympy','matplotlib','scipy','scikit-learn'
      ],

)
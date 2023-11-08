## Pytest PythonPath

- add ``__init__.py`` file in the test folder
- Using config files for Pytest: pytest.ini

For root repo
```
[pytest]
pythonpath = .
```
For multiple repo:
```
[pytest]
pythonpath = src1 src2
```

# paramnet

This project is built using poetry, but it doesn't have to be run with it as long as you set up your own virtual environment.
If not using poetry, create a virtual environment and install this directory as an editable package (`pip install -e .`) inside the virtual environment.

To train a simple classifier with 1 dimensional convolutions, run the `paramnet.run.train_conv` module.
```
$ poetry run python -m paramnet.run.train_conv path/to/bag/folder
OR
$ python -m paramnet.run.train_conv path/to/bag/folder
```
where `path/to/bag/folder` is a folder containing the bag files you want to use as training/validation data.

The bag files need to contain `/scan` and `/joystick` topics, and currently the classifier is only binary and assigns classes based on the `buttons[0]` value of the joystick messages.
It is relatively simple to change this in the code.
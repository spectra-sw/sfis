#!/bin/bash

ray start --head
serve start
python3 /home/webservice/Webservice.py

exit

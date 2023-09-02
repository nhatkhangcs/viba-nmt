#!/bin/bash

# Start the first process
./vncorenlp_service.sh &
  
# Start the second process
PYTHONPATH=./ python3 api/translation_api.py &
  
# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?


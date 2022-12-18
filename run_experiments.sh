#!/bin/bash

echo "Which dataset to run for (m)imic / (e)icu / (b) both?";
read data;
echo "What do you want to call the experiments?"
read experiment;

if [ $data == "e" ]; then
echo "Bash: Running eicu_main.py"
python eicu_main.py
elif [ $data == "m" ]; then
echo "Bash: Running mimic_main.py"
python mimic_main.py
elif [ $data == "b" ]; then
echo "Bash: Running both mimic_main.py and eicu_main.py in parallel"
python mimic_main.py &
python eicu_main.py &
wait
else
echo "$data is not either option, skipping"
exit 1
fi

echo ""
echo "Bash: Moving all experiment files into: $experiment";

mkdir -p  "evaluation/$experiment";

mv evaluation/*.csv evaluation/$experiment/
mv evaluation/*.jpeg evaluation/$experiment/
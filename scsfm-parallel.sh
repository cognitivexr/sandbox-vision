printf 'starting %d processes\n' $1
for i in $(seq "$1")
do
    ( python3 scsfm.py --source res/dataset4.mp4 --output res/scsfm$i.csv & )
done
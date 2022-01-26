rm timings.txt
for run in {1..50}; do
	./CatmarkSubdiv -c -t $1 -l $2 -f $3
	sleep 1
done
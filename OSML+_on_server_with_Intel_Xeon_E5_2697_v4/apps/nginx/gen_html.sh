#!/bin/bash

dir=$(dirname $0)
n=1024

if [ ! -f "$dir/html" ]
then
    mkdir "$dir/html"
fi

for i in {0..1000000};
do
    perl "$dir/gen_html.pl" $n > "$dir/html/$i.html"
done



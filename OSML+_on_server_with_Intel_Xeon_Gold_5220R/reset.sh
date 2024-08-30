#!/bin/sh

# kill processes
docker exec -itd bechmark_container nginx -s stop
ps -ef |grep login_integrated |awk '{print $2}'|xargs kill -9
ps -ef |grep xapian_integrated |awk '{print $2}'|xargs kill -9
ps -ef |grep moses_integrated |awk '{print $2}'|xargs kill -9
ps -ef |grep ads_integrated |awk '{print $2}'|xargs kill -9
ps -ef |grep mongo |awk '{print $2}'|xargs kill -9
ps -ef |grep java |awk '{print $2}'|xargs kill -9
ps -ef |grep img-dnn_integrated |awk '{print $2}'|xargs kill -9
ps -ef |grep spec |awk '{print $2}'|xargs kill -9
ps -ef |grep decoder |awk '{print $2}'|xargs kill -9
ps -ef |grep mttest |awk '{print $2}'|xargs kill -9
ps -ef |grep wrk |awk '{print $2}'|xargs kill -9
ps -ef |grep memcached |awk '{print $2}'|xargs kill -9
ps -ef |grep mutilate |awk '{print $2}'|xargs kill -9
ps -ef |grep top |awk '{print $2}'|xargs kill -9
ps -ef |grep pqos |awk '{print $2}'|xargs kill -9
ps -ef |grep perf |awk '{print $2}'|xargs kill -9
ps -ef |grep mysql |awk '{print $2}'|xargs kill -9
docker exec -itd bechmark_container killall node
docker exec -itd bechmark_container killall redis-server
docker ps -a | grep spirals/parsec-3.0 | awk '{print $1}' |xargs docker rm

# umount resctrl
pqos -I -R
pqos -R
pqos --iface=msr -R
umount -t resctrl /sys/fs/resctrl

# delete garbage files
rm -f $1tmp/pqos.*
rm -f $1tmp/top.*
rm -rf $1tmp/mongod_result
rm -rf $1tmp/mysql_result
docker exec -itd bechmark_container bash -c "rm -f /home/OSML_Artifact/apps/tailbench-v0.9/*/RPS_NOW"

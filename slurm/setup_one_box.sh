#!/bin/bash
# One-box Slurm for the TWC demo: controller + one compute node on this pod, 8 GPUs as GRES.
# Container-friendly: no systemd, no cgroups (proctrack/linuxproc, task/none), no accounting DB.
set -ex
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq slurm-wlm munge > /dev/null
H=$(hostname); CPUS=$(nproc); MEM=$(( $(free -m | awk '/Mem:/{print $2}') - 8192 ))
mkdir -p /etc/slurm /var/spool/slurmctld /var/spool/slurmd /var/log/slurm /run/munge
# munge = the shared-secret auth every Slurm daemon and command uses
[ -s /etc/munge/munge.key ] || { dd if=/dev/urandom bs=1 count=1024 of=/etc/munge/munge.key 2>/dev/null; }
chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge /run/munge; chmod 400 /etc/munge/munge.key
pgrep -x munged || su -s /bin/sh munge -c munged
cat > /etc/slurm/slurm.conf <<CONF
ClusterName=twc
SlurmctldHost=$H
SlurmUser=root
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
DefMemPerCPU=4096
SchedulerParameters=bf_interval=10
GresTypes=gpu
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none
JobCompType=jobcomp/filetxt
JobCompLoc=/var/log/slurm/jobcomp.log
MinJobAge=3600
ReturnToService=2
MpiDefault=none
NodeName=$H CPUs=$CPUS RealMemory=$MEM Gres=gpu:h100:8 State=UNKNOWN
PartitionName=train Nodes=$H Default=YES MaxTime=INFINITE State=UP
PartitionName=serve Nodes=$H MaxTime=INFINITE State=UP
CONF
echo "NodeName=$H Name=gpu Type=h100 File=/dev/nvidia[0-7]" > /etc/slurm/gres.conf
pkill -x slurmctld || true; pkill -x slurmd || true; sleep 1
slurmctld; slurmd
sleep 3
sinfo; scontrol show node $H | egrep "Gres|State|CPUTot|RealMemory"

#!/bin/sh
# THIS IS AN EXAMPLE PIPELINE
# PLEASE ADJUST TO YOUR NEEDS
# CHECK THE SYSTEM SET UP

# ACTIVATE ENV AND SAVE SOURCE AND EXEC PATHS
source /work/lpdi/users/hartevel/venv/pytorch_cpu_gcc54/bin/activate && export PYTHONPATH=''
export TBR="/work/lpdi/users/hartevel/bin/topoGoCurvy/topogocurvy/minitopobuilder/toporeverse"
export DSSP="/work/upcorreia/bin/sequence/dssp"
export CREATEPDS="/work/lpdi/bin/utils/createPDS"
export MASTERMATCH="/work/lpdi/bin/utils/master"
export TMALING="/work/lpdi/bin/TMalign"

# TOPLOGY IS THE TARGET ARCHITECTURE CORRECTLY CONNECTED
# ARCHITECTURE HAS ALL ELEMENTS SORTED!!!!
# BY DEFAULT RUNS ON SLURM CLUSTER, NOT RECOMMEND TO USE OTHERWISE
# UNLESS MASTER DATABASE IS SMALL
TPLG='A3E3.A2E5.B2E5.B1E3.A1E3.B3E5.B4E3'
ARCH='A1E3.A2E5.A3E3.B1E3.B2E5.B3E5.B4E3'

# PREPARE DATABASE IF NEEDED
# THIS CAN BE IMPORTANT TO SAVE TIME AND DISK SPACE
#echo "0. clean sets..."
#python ${TBR}/filter_by_scope.py -m /work/lpdi/databases/master_scope2020/pds_list -p scope_ab -l c. d. -s /work/lpdi/databases/master_scope2020/dir.des.scope.2.06-stable.txt

# RUN THE MAIN NODE
echo "1. making sketches..."
python ${TBR}/make_sketches.py -t ${TPLG} -a ${ARCH}
echo "2. run master..."
python ${TBR}/run_master.py -t ${TPLG} -a ${ARCH} -l ./scope_beta.xfilter --master ${MASTERMATCH} --createpds ${CREATEPDS} --rmsd_cutoff 2.5
echo "3. clean master..."
python ${TBR}/clean_master.py -t ${TPLG} -a ${ARCH} --slurm --dssp ${DSSP}
echo "4. create final sketches..."
python ${TBR}/create_final_sketches.py -t ${TPLG} -a ${ARCH}
#echo "5. cleaning up..." # now inside create_final_sketches.py
#rm -r ./${ARCH}/okforms/*_results
echo "5. sketch processing01..."
python ${TBR}/process_sketches.py -t ${TPLG} -a ${ARCH} --slurm
echo "6. run TM alignments..."
python ${TBR}/calculate_tmscore.py -t ${TPLG} -a ${ARCH} -e ${TMALING}
echo "7. all vs. all TM alignments..."
python ${TBR}/make_tmscore_matrix.py -t ${TPLG} -a ${ARCH} -e ${TMALING}
echo "Done"


ssh fkorbel@max-login.mdc-berlin.net
#password: Huhnmasterarbeit97#

#Create interactive computing session
qrsh

#Activate conda environment on computing node
conda activate integrated_grads

#Create a link to the jupyter notebook; to open in browser on local machine
jupyter notebook --no-browser



#new terminal on local machine to create tunnel from local to jupyter notebook on server
ssh -N -L 127.0.0.1:8888:127.0.0.1:8888 -J fkorbel@max-login.mdc-berlin.net fkorbel@<computing-node>.mdc-berlin.net
#password: Huhnmasterarbeit97#

#Copy Link into browser on local machine
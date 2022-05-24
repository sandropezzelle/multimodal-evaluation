import numpy

models = ['LXMERT','ViLBERT','VisualBERT','UNITER']

model = models[3]
targmodel = model.lower()

resfile = './'+str(targmodel)+'/correlations_layers_lang.txt'

with open(resfile, 'r') as myf:
    content = myf.readlines()
    layers = []
    for line in content:
        lines = line.strip()
        layer = lines.split(',')[1]
        if layer not in layers:
            layers.append(layer)
    for l in layers:
        mystr = str(model)+' '+'('+str(l)+')'+'\t'+'&'+' '+'LV'+'\t'+'&'
        mydict = {'RG65': [], 'WORDSIM353': [], 'SIMLEX999': [], 'MEN':[], 'SIMVERB3500': []}
        for line in content:
            lines = line.strip()
            layer = lines.split(',')[1]
            if layer == l:
                # round(answer, 2)
                corr = round(float(lines.split(',')[3]),4)
                bench = str(lines.split(',')[0])
                mydict[bench] = corr
        mystr1 = mystr+' '+str(mydict['RG65'])+'\t'+'&'+' '+str(mydict['WORDSIM353'])+'\t'+'&'+' '+str(mydict['SIMLEX999'])+'\t'+'&'+' '+str(mydict['MEN'])+'\t'+'&'+' '+str(mydict['SIMVERB3500'])+' '+'\\\\'
        print(mystr1)

# Vokenization** (2)         & L(V)             & 0.8183        & 0.6544       &     0.4420       &   0.7748      & 0.3026 \\ 

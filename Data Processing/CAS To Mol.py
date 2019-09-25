import urllib
import re
import string
import os
from rdkit import Chem

****************************************************************************************

def GetMolFromCAS(casid=""):
    """
    Downloading the molecules from http://www.chemnet.com/cas/ by CAS ID (casid).
    if you want to use this function, you must be install pybel.
    """
    import pybel
    casid=string.strip(casid)
    localfile=urllib.urlopen('http://www.chemnet.com/cas/supplier.cgi?terms='+casid+'&l=&exact=dict')
    temp=localfile.readlines()
    for i in temp:
        if re.findall('InChI=',i)==['InChI=']:
            k=i.split('    <td align="left">')
            kk=k[1].split('</td>\r\n')
            if kk[0][0:5]=="InChI":
                res=kk[0]    
            else:
                res="None"
    localfile.close()
    mol=pybel.readstring('inchi',string.strip(res))
    smile=mol.write('smi')
    return string.strip(smile)

GetMolFromCAS(64-19-7)

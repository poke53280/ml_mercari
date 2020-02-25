

from mp4_frames import get_output_dir

from pathlib import Path
import pandas as pd

input_dir = get_output_dir()


l_files = list (sorted(input_dir.iterdir()))

l_name = []
l_part = []

for x in l_files:
    zTxt = x.stem

    l_txt = zTxt.split("_")

    if len (l_txt) != 3:
        print(f"Warning: Cannot parse file : {zTxt}")
        continue

    iPart = l_txt[1]
    zName = l_txt[2]

    l_part.append(iPart)
    l_name.append(zName)

df = pd.DataFrame({'iPart': l_part, 'name': l_name})

print (df.shape)

df.to_pickle(input_dir / "all_originals.pkl")

################

df = pd.read_pickle(input_dir / "all_originals.pkl")

# Row 1

a = "zqrz"
a = "ivsk"
a = "vedxk"
a = "dvqlll"
a = "kewxt"
a = "bfofp"
a = "bznv"
a = "fnutl"
a = "dftsw"
a = "sjkig"

###############################

# Row 2
a = "dvcl"
a = "bkuo"
a = "cmley"
a = "cifsg"
a = "bamub"
a = "llptz"
a = "dkfyv"
a = "ibcoee"
a = "kjkvq"
a = "feicji"

# Row 3

a = "jdtkz"
a = "kyzj"
a = "hwyh"
a = "avpuw"
a = "qqkvod"
a = "csaui"
a = "jxreja"
a = "fdip"
a = "svtju"
a = "klqfp"



#############################################################################3
#  Row4

a = "cbgry"
a = "ahzfq"
a = "cflko"
a = "fcqoj"
a = "imsac"
a = "hxlr"
a = "ydgt"
a = "ezmvb"
a = "bdhuo"
a = "pkhkw"


# Row 5

a = "clby"
a = "fhncz"
a = "bbbat"
a = "vdmyz"
a = "fkids"
a = "baxps"
a = "bnsxu"
a = "avdvt"
a = "cwbgz"
a = "fltays"

# Row 6

a = "cbxvj"
a = "spsev"
a = "dcnuh"
a = "hksbf"
a = "ndfbax"
a = "bciyp"
a = "pwppr"
a = "lxmkq"
a = "dstib"
a = "okmqu"

# Row 7

a = "jcxgvee"
a = "biytpp"
a = "hicjuu"
a = "pzjry"
a = "xkmyy"
a = "obhwst"
a = "pqhou"
a = "lsdcd"
a = "nepiit"
a = "tvfyff"

# Row 8 and 9
c("ealzr")
c("eagqa")
c("lpttjs")
c("slytjs")
c("cbiwg")
c("zxafp")
c("ijnfp")
c("gxnfk")
c("fecys")
c("glamn")

c("lccey")
c("bdwpy")
c("mvdxf")
c("offdm")
c("qyaiav")
c("sqgvt")
c("fllcc")
c("ifgqd")
c("nrknr")
c("mudvg")


# Row 10 and 11
c("ljkwq")
c("tnjro")
c("csluw")
c("jqbue")
c("jsyvj")
c("tyvne")
c("knbod")
c("eypz")
c("clquf")
c("pgwvp")

c("mdwt")
c("npvws")
c("ghptd")
c("zurfg")
c("agbav")
c("itmnw")
c("ensqj")
c("qvxuz")
c("cgoqh")
c("yvhit")


c("mxmqh")
c("rttxr")
c("djxdy")
c("erufb")
c("soydh")
c("bypcsh")
c("bmpryt")
c("jemwfz")
c("yjbob")
c("qfalft")

c("hgtsz")
c("sllgua")
c("llcbay")
c("vfqgf")
c("zeyxy")
c("dtfqod")
c("mejgv")
c("ofosax")
c("oxsmx")
c("whldc")

c("tthzw")
c("sgjbak")
c("dekub")
c("nperme")
c("heoufz")
c("pbfrp")
c("qlkzc")
c("tbiya")
c("isnbk")
c("vdqlw")


c("hlgjnu")
c("xdkrcnt")
c("xsrqm")
c("ipowm")
c("nvldb")
c("nfimd")
c("ebkatw")
c("gghhm")
c("brlra")
c("qaaqq")

c("nvmnv")
c("zfamzp")
c("kfqrag")
c("otfnn")
c("snjcin")
c("myewu")
# Missing !!!! c("ked")
c("xperi")
c("jnrb")
c("sevxhi")

c("esqv")
c("uvhxg")
c("txceg")
c("nqdrt")
c("ohnon")
c("rwndz")
c("yknne")
c("ilkbv")
c("mkigb")
c("uqzbw")


c("lusvm")
c("fksks")
c("dnocq")
c("fpmab")
c("diydig")
c("oafhp")
c("iwthi")
c("cntnc")
c("kcqrjd")
c("espkk")

c("cxwvz")
c("jyxdp")
c("vsvwro")
c("ochzcz")
c("wyhnv")
c("wqjre")
c("urqnlof")
c("hrkwa")
c("lgmxcn")
c("yrrpp")

c("dryto")
c("ticjnw")
c("tworzx")
c("arrpq")
c("ynfgxe")
c("uyzvp")
c("kosmf")
c("sqyycl")
c("ieoerv")
c("xqwzan")

c("mumpse")
c("qorwh")
c("osqruu")
c("hpura")
c("uchzxj")
c("obkcg")
c("xvkcb")
c("gpcwa")
c("ftfemk")
c("qjvorp")

c("bzfhk")
c("ukhtg")
c("wzetgj")
c("uiwjv")
c("qouib")
c("cgjemt")
c("vhtds")
c("kvhby")
c("sahwx")
c("nqjzed")

c("hbqez")
c("nzhlw")
c("xqvzc")
c("fmutb")
c("rdfmib")
c("clekum")
c("euyynv")
c("vdaehw")
c("jwpyc")
c("eejln")

c("kzbwt")
c("zbgss")
c("mnijii")
c("akyive")
c("kztmw")
c("amfly")
c("epzlf")
c("gpnkd")
c("gochx")
c("afxat")

c("jnuym")
c("mnmpf")
c("whhnuq")
c("jhmliu")
c("psgcme")
c("xdqbae")
c("lrwhd")
c("clpxc")
c("fmsxom")
c("lndhy")

c("tolsk")
c("zrgvt")
c("stdavr")
c("ogpsz")
c("przmou")
c("fongi")
c("qmtchq")
c("vwuie")
c("aebkht")
c("gkmsvn")

c("kxzyc")
c("iwhkix")
c("qntmam")
c("ewuhue")
c("eftdhq")
c("kbulxc")
c("nrerlt")
c("vjkiyq")
c("rtpuwt")
c("ljkoj")

# MISSING c("fwmqk")
c("kipkhe")
# MISSINGc("wyay"
c("zpqe")
c("naevg")
c("oboxm")
c("pntqu")
c("lctglt")
c("yaivzm")
c("gotvgc")

c("vqbbw")
c("ibqqieq")
c("jhlnj")
c("pzlfxm")
c("cxptp")
c("oybwz")
c("rbhwa")
c("njwimw")
c("xeuhr")
c("filuu")

c("lqntrt")
c("axdeps")
c("vftcid")
c("qvdkibr")
c("wpzerce")
c("upbgva")
c("dmmua")
c("xbanya")
c("osnazd")
c("qockie")

def c(txt):
    print (df[df.name.str.startswith(txt)])
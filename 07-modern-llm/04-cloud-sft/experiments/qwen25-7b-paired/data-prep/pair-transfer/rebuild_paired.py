"""Rebuild the exact 9k treatment dataset from the frozen 9k control; Python standard library."""
import argparse
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import sys
import tempfile
import zipfile

BASE='e314fe11f0a578009e67809ec7d67e3c72484756a25e6df50825eba8f41a279e'
TARGET='911843fd787d5366efc63fd73af290ca40f37cc12da6365ccca898b34afc6965'

def patch_bytes():
    with zipfile.ZipFile(sys.argv[0]) as bundle:
        return bundle.read('delta.xz')

def restore(value,refs):
    if isinstance(value,int):
        return refs[value]
    return ''.join(x if isinstance(x,str) else refs[x[0]][x[1]:x[1]+x[2]] for x in value)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control',type=Path,help='original train.jsonl with SHA-256 '+BASE)
    parser.add_argument('output',type=Path,help='write reconstructed paired train JSONL to this path')
    args=parser.parse_args()
    raw=args.control.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BASE:
        parser.error('input does not match the frozen control SHA-256')
    if args.output.exists():
        if args.output.is_file() and hashlib.sha256(args.output.read_bytes()).hexdigest()==TARGET:
            print(TARGET,args.output,'(already verified)');return
        parser.error('output already exists with different contents; choose a new filename')
    lines=raw.splitlines(keepends=True)
    if len(lines)!=9000:raise ValueError('unexpected baseline row count')
    suffixes,records=json.loads(lzma.decompress(base64.b85decode(
    b'{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;SWST)m;EB;Doqtd>1tycbJHK5V5{<i^})d@)5=!H)I~Plju7Zo|hvKCz{mNDkni7yv{%B'
    b';eVzMdV${kezVi|&Cq{irvR5sa^H*3%V0ybt%VW+wq2#PoXm?b2J}|dOW#MftcYGDP%wfSsY5+RbEHyWEq8}dvpf6G9)`Z_k&~9Q'
    b'L4ra)M`<~lfrJ){PVUu9$M0#uHslEok|usB>5+Y6S67&iPmepkCk0_-Wh>agf+(mOF$$68ak%4wsmC`X*Fl~_(LjXjZlj;qx402~'
    b'wQLUu`%1xO|JjW!ELP0Z_pepnJa4^#+-#BV7V$n5!B_{t9_|@>N`R=e{L|Kc*=>P5Ig>rX%!JsEB_pogVPT}Wanqc7SYjNW{X;J@'
    b's|fb>0al=1!9bZPa^6^cjnBRB7m%GF>8l=zQGMnUV;-erujSCERnZ6f%E!|n5K{v8wde$tu`>m&5M11#Wj%STO2Q>*=gXQ3mk&nr'
    b'!5yOcwH`KT&5F7G)4v>+=pAMS60s+2mm;o&@(~e$f0Uet+tWwvz^@n)O_!{<Lto2$sFVCND{$Y|iz+kk)|TmqfHuFxaH|nf=$ZdK'
    b'<AVKKD(mavG+nKwhR5w~NdELLN-T4MtuR{^WG)y(qDXT*<B2y)BpzAuo$QZ@q{TP`ed_rw3=Xd(v*Q8z-T61ShWv$WFlV_9)AwG>'
    b'rJn*oUq@4(hjqdUPL{z`9O?*oN3bDdzf9U%MLST-RmtJo{mG9s9Qn$oWW#ihXn18Z9><wL2TFPSlsol<^f>}6gD>wv62K2_g_+D_'
    b'Itf$<fkon;aJ6L=FW<&nX7grR58NXh==4iW2VCyb>Ey<s<T-oxX9`E`RwJD1@D-VK)YVB*k^Q=h@C;Jw0@8BvaXR4;0@)i36DL=('
    b'(8hV`Eg*WJ8b)1iB5vbf1at7Cip9So3sLH$@x_PARW${w>oQ@^Zs1!8@ka)J(d$K}8~5g_oSPgIs^fk0w!TX8_N(-UW#CTGgnQ(}'
    b'f1CEXv~VXM!JJ{u;tBe%_015_bju7^8El7Hb^|rzLU9yQQrVG&o~t<Yo_%k!ICzR+pMIgtb!tZO2L|@DmF;r*UtRoiQBvqh+HG@<'
    b'Qnvwaf>>7e`1WjD5xK9P6BVKCvSf=7BI%|@(X!o}w}GtGRq}WxeddJ?T)(#lGKA>Ebi$#l8OirQilMu;EqVtB#1SRODDJT2ncmBo'
    b'^BJa3sp;njZ@Icb=z0825I9oh0d1mGn-e4B3gA%;yq05ay60DzZCT_L)|7j<T_c)kD%>FF8UAX^Ny?UmHy&Dt1!l+Wy+a>Y_al|1'
    b'BH_)6%!hSj{B4T+w`8ab?Xs=&J^a*XKx_AhKMP_ovoO;2l&6As3o%gw&UG+t><hDeW%bOm{bb>vJ05z8GMdof_mEtUN|gTL-9xET'
    b'F6->_Z5#y5W<s_?c>8e-6X<Ec-EJ<6+|p?>jZ`@#VgNLc?w&DQh&OSN?dF(#41g)!-gC}le&mXw(&r4kL?fz9AD6~L7TX-mfD7TD'
    b'5pqQWldji$H$bL531$aK{fdf6Yg8fy<z9m_?(`*E0!KJ!1VrtnZ8O;nGzYVI?~ulC0{<L0P&~s?5O5T8+f+ICEmh!$O$NSz(wxH('
    b'@LM~k4QXQReW~a4wsuv);AFwJatg_XZ-z8}Qg?daxx%ma{d(lbcQ%K7idIz`{^woXp3*6EMyr93H7}_lx^L;UF>7vIB0{-b`!OH@'
    b'BG?lNk4flkc{SJ$#y2VYib$EtE!94(tov{)tS6PhI&aDB*(5soDZ*cKjhaX`uTMi4gBn|%yI#{!<TedbeZkY~JZwXRj(ynAu()cl'
    b'ki@7!1hb|@uNe11m`LU51ffjEQ51j4MjE<&*4mn%MUaOX)aaVb-N>FhD`V)0owr;M$3x@;AX;;N&p{#y`@!un`PP1f5gNppJW#Cd'
    b'$q7!e#9y^Nm40mfW8w@#vu-5&<F)mbx1eo?0Kc>i*G+Db8fep&XZOm*gEc|fv~uQF?<HJRza|_Wz^Q|mhcXbyR})NX|04>IN;P)7'
    b'+;GqW%K=pBqxL4MQY!t(GojNYwdB*Z(&up+Rnb;zGE_AOSr5<TzLB;Ach>WvPz?h_O<hg3BV0>Tp>DjnW`<Jm9HM;MFOYaLn;>Eo'
    b'vR5!|@96;LN28!()n@Rl6xnnWnmuA)rNFgk9*QW6#C!!P@x9yaM=|opxW6WYCmo+qXh#TGKXSWf%H?28s(=-B&}O(DRBObm=$FLG'
    b'ec-!r>PR3a@05@?O~z1%Abu)qWx*S&%-&ciJ4pTfDhn>sx_Emkup7|(lMNKn##?8Aoy?@>b8kF4h5Tb23G0n^qT)K`x}Gph0g#1K'
    b'UPl)@18cZ_*2Y9a7-oCU9sL4W`zliX9qh&X*A14Vsc1cxmO*`EM>zBL1EQ44E1?rOIXv7_h4GtfD~&)>;Y?z`ppM3(6ov;vBS#=2'
    b'mkph7b;>i#Jht5q73*aivs7@%seP!B7L6i#8RGv#02_g~O<HZX_B^qPJR0opwfBrj23n(@LH(^R99*C-owpoj*eHrgwy(<bqVH>E'
    b'^yWWh7oEvMNG0_lz29CROgMCm{mDVr{gU$O9dSxsWald%urUG0;vmG44_KFVE=-|tU9(H@ImuI$xmo6;qq)!kg)?rF{_&-s^qx^D'
    b'E;L~m9;x5J2prPk4wmU>T<}P`^>rL$iI_o*b;X|g3YDDhd2$quL?%WC2-_D(I=N(?d#?~gj>H9Dqys|`ujDXA>(bpTW=9D272?_i'
    b'L<XD;jh8T$9G+3*n&g~$wPX|NE%SckR{9bmCnd4awSbIt&&SWXxnQEH_L3**7<poT*4k!H2Du-TL3fbc#=a3upu{$U=)X9y091~w'
    b'FhzNORd}6l8;#G6MV0D~<xcFJtCeqe*%Hn@FS5qPt<Es{E7>1{ouIeLG{A-uWQ*VySLL+NL8?qtW+v5$ho~S-n^1R%A?rEwz?~2>'
    b'akYW(y{ksif{GwJy&F5$b=SSA+P01A9q@9r3yWUIh8Q+0R8}BVzsfAD9VqZT_EG5D^`uf0nnVK%tPxjp^KE9>n59_)B_HtnVgugx'
    b'>myEGy!A<TheNji^8>js^3x*9RySdT#KN>C|Cx~1@^|G19(f%fNjl8cL}5HK<mwa|$J8zC*@+GxR%N^tG6vR)>`EHDq&-85fmIE*'
    b'Q}#`C#|uI?s9!CY&KD<>Sa4A7f|Zs3O#@0_b!n=s94Hs%xVLz9D8Zsi^^hSTvf}MwId-$%SE&ofE>D#ZU0PqyOvONVs?x0<7E08$'
    b'+d#wbUJzuKp)>>S-ii3NvWdndd8a2Ol-9ivPrnGJ4OQ4Glb#eJ)zoc68!SX{4dibd$Mt1xnOKSzEV`%gUtwUbm!oTdI6?gUMfSkX'
    b'G=hg;^zqCmn0!?jz@!!~%<{}HNh}7cTB5pG>jz>f*<ee*Irh<sg37Sr!>3OCiNhDcREq&VzePG|b}W*YV{ekcvvaz-7)IEjG?P=L'
    b'{js;s6BId`h$&d8O(<pZm>h@#n^h-|^N38`;z@91$9=F3MZ(49F+|UW)zR!2xD~FbVdQJG(Ky2}4Ka)36m-ycRG?Yb_-iGNExY7p'
    b'>LJAE|K)@YzF6SA9kp>Ar{Zkyv78iHPFros-r)v8p-2+yEOO~wXTUg#5#oIS&x)#Sv}!RdIpkwCi5gKqW)ZKz#m~LvPR7|qE8=FI'
    b'*qD-wi`k#k;Q(#NWcScy$W}=GZ~btYm?-H=H@X~hoqG8mV9!vUX+3{{`f?!sIgEYiJybmSy0|Q3-ARj*pPC3sC&SoHuTt@bjZN;P'
    b'u6m9$MoTy00?6U~ckjPi$^bO8^)PBb_)WV_?*1<6{}G$o0%?rOrgRK1agM+^{Lqv9!cyKR>1&cjmJLSV<wZ(;UnQ$MS)!9s<uAD_'
    b'Gu*Esxu4h2RU{p_uzqK*Qd(qOOQy^x4$CTwt}q40ENXQ&4etz4Vh?vFpAXHd@?STO>8U}eHVTG2#Vd%k*g~2<3&hFHl4LZ4YPl)5'
    b'I3dr6{QYJ!*a&nqfV6iBU)RS!Oas~yEW#+vxs<AvCk0m)%Mqk_qg_KOwi=ngG3T2K=z^d1e|l2GZhqU<eHR^nmT&{m%Kkkr1qIb8'
    b'ft3{w!Us20%aGU1zm)qw+d#<63}Lnm$1*Ib+~YtY&0M>Pt=zA1&Kq&djTlNRB$SD!W;J3|)8)|>2w61w39SP!c<(+pJ-S*oh<a)9'
    b'-+AGy)o|82rU(!$X{-h>4CBPi2K|m{faGr)<ux`putSG1i*)M3v#d71fI^<RU$uvtBFP!paHW-EWTiE$xaaQnI2Q$R8}tz?hx5w@'
    b'-b%Z^*EW7s$ZS=Im$We^Q~Jo<;UNHXdCwr(HiHtirJ&5sk)&TS(kQ~5DyeuD951VtcT8v8)aExWzV?Pw^i7cIUIkCw(KM;On|R?z'
    b'h9RH50#A4QxQ1u5K{KE)%J4u%x8VC}*}=rbjy}WRX5+>~96r+F($=<K0fH^$p=A4QUr$n;4VI+ih_^9p3yndq!@faWc~wuV;v~^M'
    b'Y+6&Gj<}pX*h!N~sHY`BKnn+VMzc%*OSsSCqUG}D-=1xt2<b-}dqASMqJa+J?ytu^B1}EBnc+;(=<yWUZ<av1!fR@U9ipz-%plaN'
    b'|N1M59Nc;5P8xHTB0n>Vn`~$j{OGgI<8KcG-#Oyd0Oxm?(l4;U=Pjt>31&}W6^^lWNAZmcfxF~lMRz@Th39}Vi5zbA8vgouEg2*7'
    b'!PS&WQf(-}ucqhxzj0{o5fj&4d4t|On_AnwFR`UgCas2GSSGCBCF|v23m)4>f|&&Ov>K_?urAdZx`)D$5f7=QZbh?_V;rJ6B^ZQ='
    b'F&V0vF}DvJ@0cdn1`cSqd6z*H*H6<j{j}Ih(HS8G;2$&?%9vTR)`+74i6jk&R-G${PAzdBqF-5bdk@AL@WtKMKw#a1W)OuuB3Qw!'
    b'UJH{|M;mtFQWL*O)j+gPD=WBrG67F<n_R))$e4vlp5ezmM05M94kPAZC3A_-`E9*QbSbArmd~nJ;5%@!@yTnnxw+?Ac5uJv1s3C1'
    b'BPpw52B(8%t)74<+L}qYY&<ils3q)C{}>2`0?+}&d$c?cSY!8Au1(!IxJ|jDFc6K9)=bjJZ=Sdb`R1&}%55t_c`Vaw*)s5m+Gy)#'
    b'eHAMT7^PyeB(0B7uQg`mGkV>D;FXHy$#1l041TcDjS&>CSdCYG!R5pL2$KR}2RMn<EZvB6%)gOnFV^Q3Uh^If>|_>sHH|oQTReS('
    b'V;=N(!rOYIn9HUzY(~3FC8HQg;+?c5R6aUkFsyyv*H>45k#5?qWGE}QR-SAITEU(e$>;wUR*uNGhG2io|5pBvz>Szs9R<U5)QfWp'
    b'6NdVnp(xCv9ROVLP7?`kdt8(Jm@!McE$dW%3v2Q#jRuf6f<B@CIO3?2$EF$gQp|wI`MrFdaz7AFqWmD_b}GT#T8B`5$k32{bb!ot'
    b'%IR%h;^A^w)~f!Q@-6&0b2ohbV|UJ{p^rE}+Xk0}5l@d}0z!+TtwI)&jTN)O<aiq2b4^^%er?maM<>=^yF(nCV|*8KuX1Qd8fm@b'
    b'5MFvlF>eXaJ(ALjy)oq~dueJOoxH~nf*geCp;Nya>tz(|M!Z3%=An|`FdmlGN`}|@P1qurW`*h4ftRqJV1&`sQG$}fD`)?km0{E;'
    b'&}k<BdR~G_)q;pncjn<SpO;UE7<!Y$&*$(!@Vy4^mA}{YL)s73r0(xK3t!HuL|ZQt6A{1Wd~YVCj0XtlMEVBIw0AX?AVqlu*e{Go'
    b'YwogH4=_#GfyUBdFjM&DUF3Hz2@S)maAKr{3P&%Iwpnrw>Eqa=JTwdbDHEB<nG6-2Sn-BjE3N9uuapOk)e!Nqe=J&YFJMB0wk}h&'
    b'+<#k-RUr!4pbXfbses9l<lFoWHM2`P^U2M$C-I1jh>2Sym7ycPHop;+qkC6y`Y8l6VU~cpLb~IIC+dq7yxde)l=CsV5T>Y<W<7%S'
    b'*q|8DqRzs=jM<~gQ1J&R%nyg5@!hT4?raPQ*g4nTVYPTO%XOltdHD5uGXMc$1eNVrm33w7$Ef^Mc6v%^SKhT=rVsZ`-=@j|1nox3'
    b'?V|bRYTUpe3`1|Ud}gG{FIV}AzDpUwV^IO;sv4Z*V(%uz$X5!U)wLR3#M-GspJU|8w<M+s>(^Uknu<er{N?k}N7edC>zR@~<o|w+'
    b'sKBHVK+(shD8qmJwo^sMm<@S@9g&;&1RU=K^k<2x?pTzW?e^7Pj5q;9L>@pA(+sg+8`r!MFr{U~xKbZtCM_}!mhS1-WtG^bgXDZ{'
    b'l9mEsNUc3q1&f8=x2V&HF(x4)ucD1DuNiE2@XygrQy>Mbb5^+h;qMJ0Vm<-nQ}ZH|LJ9nCfVv~kr3LIk`qt#OCi&%2!GN|=kK5ww'
    b'$jHlWz|SA6OxZ7_ArAA|2Zn2R422MzWh7lwl!!XDLaq_b=4~C62g_sdcM{_uHT7-$Up8OOK!|i`+vRJyyweC7g7LG6_k$1#4aPxg'
    b'IWZkaxZ2&NL2Wu4^OOxW#S5Sfyw&%3BIS3D<@W+D3Sw0LNSUo$Npav#E#^Ol@SU(a7MtABzHWe2j1~ZImX3=VK=E84Z-(o3ablD_'
    b'Y&1SZ+d*1!gL!#{kKbuAjD}py-V0kN>hR9Y5K0d?aSyyL7Fn8T5=#z<t~-s9*i_(U#t>kx_c`Un_(`B^{@+JlN8bQ1u5NqW*C=P2'
    b'4CIuSqv^TGTdONU4uvbG*L7Z>wej@|nXlTXKTCw^`wD8^BEK$~4iYlOD4mvEGrYg1(x<;}`Ox@Bg+ubnX-qj7sR;K9KqQ)Jy~4LV'
    b'sMvqd7q>Nk`$5hu(Gj}eA(5$1!tCimS6-?e3-^i8%a`%pH3G}}pMe(?$iDa;lA%Cx^uHlC;+|;DYtAR_O*)Mf9Z}Eaf_kLYRE1^c'
    b'<EKskw(R%iZfpRxf$wB_F$tBjRwd0N?ZsO-5MNw?wdT4;1UT)wRpq1IdQ1Jx;{CN`izU61G{b@atAf}jpkvPF%(h#1OaPts!g-2F'
    b'mNZ7nFBW`Yl5Ivzj=G|?79?I9Kcz3{u1jLZDY&vPnyJRrh@q2jSsjvkuR^xNx#DG7zo;Lq<uM+hx52~99*zp#57!FL=mIjG24${9'
    b'V6eXThrkO2t|w#U&kk9Vlu)vy_m85p<sC3!_O6Hqr0Cr>hgiq*!GGyWfYUBxXERBe3VfQXzA%=wcI`({<#jz*C~fi_$DuXciN2Pm'
    b'v<V-L#o({C8~V?wTAf2YX2qT|4g_<gVZ`KI`%<A?0vW+Y7&`Z7$LBf^+wNUV%d0$&Xv^|}P6PeT(!os%I9f+A#|mSVVymV<HUv+('
    b'5P?>r7#-JStCkMuz;BCp#>H#)qEHQP+Ogm5*~X|VQ0dCA2W2|o-mjkkck_0j8OU`-Ms~1;!A}+u`gN~XDPg>2gv&=T7P(CKSiR1K'
    b')W;4{rWbF1`z^#RF>u05=FoO16dd`CUe-Z4wjn%L@~&O}`PBC8nO4GAgmPBHBUhWMpkKyd&iyfH9Ln4iv;|wzVPG6FFwX9;`w|46'
    b'l0ML0j=AMZ1v9>k%L5K<)gfvPo)Hp)FxAnzp^Ha*ULeHUQtv=RjBF^Zr3H1bxE-8wDBMAcyx`yHz=WEkPDxit5M)B<vM^}YZkkF2'
    b'wwUFl1|~Q&b|N(C4RsKeUe5c_JbNF=T1Kn~yjhdQipbtDbNUeqi*|K#(}Q^G_=X`NdUzd3+%hjKUtb{nAo~S1*Z#XlxDRno_gYGr'
    b'g_E>^segdbJXhGuXPL%TH+E@h;~En&jc)qCn3V!9DS3EV5uhdSE9fb`zMLYmKi1xr;B0>lbT21sbzlB-nG*qbGUNukU3OwZAeu?b'
    b';-I`!)1PkydD7a!2s^qA=VTwW2J0ZB&77M{ZSnWuX~isIiJ75OdMYKh5N;!pLp`*Ciz=K<UvDqXk|$f{+UL=!nb@M0yqxohs{8vK'
    b'Q3P?KCq+)G;h8U5l4}4;u1qjcu2T#-Gb7DPuJlKGZr^)<p*80|E!w19(~n4SV^+TI2@G$`CO~)I4U(P`x;GWVb`eMJ$3=gdgWX!@'
    b'>9tn(f)h7xd>BYIDtG=}P&os2xb7}6n5XQ;9<BsN^l?}lEKT!gb3)7HI6+Lzz=}670N?Y4+?zi_5iXWrdC3eq_P&nOd&2rUaMom+'
    b'wd8aD6Pk`xXp9cSS|9Y*Hh;5+BuXylM*#8&6ei?SxAu@@9!F%0+=nG@WS-7p;&WP~<+dVj_Q0qWz(tSx#$P%|wh{19jP0#pk!O91'
    b'zz7WNh0Ibo5e9`t6+<vW=>JBK#WCCn=;f~QB9c`g<Tv=*S{3N>sxOrpQe59>Uw3BNGb7lLarau_)h<1FlYh0mD***8j~A~0<CdHY'
    b'<u>ti5&hHL&@@YSlQaqD<$NE=Vd)!H5Emp9K`>UqAe5n3*24R0(;QH=7}!J6cPI#Hf{LremgCrzZEKn!ln_L#9?|`UuWF&SU;&I@'
    b'hI#L4c>=VQm;~SUBJeRfciZosOvB|A#She<3e3xXvIWo8J^Y$3_6t&SCDsN3iJ|jS06yY<Mc;v!Qyi%Gg5kyaZs?Jc-mL=U8a*ip'
    b'C>;XsAQ=%_&0?~DdAAv`F31;v$8#aZBEE|xx?iC~)NQB*UbNF;Vxt1N{Ye=j;=5MLOdpCyYF_*GnWge_+d@*7F6XDEu#-Wnf9ovt'
    b'?jbR5n%*HRydf33N=*&m0{y7gxk|CjZ(V{>1<Sdwq$AM4crHxWknR4o?PW1qW_7Wzc5>Tvu`tO%CSl9Ff=})=E)0*Yg9)pA4#-o4'
    b'GlZfa#4H%B3_=Q`WW9f2I;A^UwME6HcUbI%b%@LU9e<*}{1k1#PO8(o=a-&;1o%U1Qt+iF-fV8#cH>FRA{7}8jO9!iYfy3;+-T(K'
    b'6eCJa@0j7h-Gr)%V{l$4*sHjL&<ktASPGuPlIb&9S4`D`wGymiTdh@u#yzib>)Wxb9%q6Uo@)t1dt24A$s}|^h0Fb$TVjfkU40%+'
    b'!u0bp5cJ5sUf|g#Nx!a&EN2~#{E_;c%EDLoeWWN{+B0?P_8s6AK4x2uMPVM3u}EXKU@qKVyh;b{?&Dt#R&1jHjFe#0cOl~#Y<c41'
    b'bhnW_o%Ix~9`KbD2K6f4d7A$2#xC@bob0Y`dHvc+OfU4+wcl18RrLK|y1~DOlctpA|Dm??V7diGJ`EgQhj-;IGiR-HzqO^|itA4w'
    b'xJGf^e|JFTuPN7&nPcaw)F^u(GJ}@HP9xHTkuM{+6`p3x#=OuguD=9$^Z0~c>3DVAa+%NsRMJU6Y+!~Ugx?c!yd1xM=N$(dhYvvL'
    b'qrYMEdp}6U?VfJ9g*Xz1*z^?xL5SA-V|0jSM97>H##PJ%mFYp&$+pa`@-E2QMk-}DQ0-8@iD^s{;z)HPn<pQmV=#)v^=9g_z)cl_'
    b'PM<xz9*zIFaL*VFfK0Yq5@~0$>TE$hlcqOnm`jXzn?^;No49N(cBSErKrBshg3&1(nVy4IQ#t&p8m|Bv+pJ2w!&=j2QC;zZRi&dy'
    b'c`>8>1rMExllxU*%0HwcK7rHTmp?(Ld3($@wq~hm$=tW<9Srgu{@haan}<e)CpNIAz)4Wm1mNQc50(JHTI$P|v=k5o(pS<g*?5(w'
    b'!{+5rk6k;f!K^2!P(%N(QOsX)1`WB5P+L9RJ~B48J(7sHzSCU5(_<O=#%=m^CoV)}D%d2D!Ql=hL4QNo4Q(%9lSc8d9&>t+{*4UC'
    b'=oOxZNUiA}L5*gU_X)ah!e-04BhvN48mYXm4Graopv=$;SZ_mXT^gM5v|P6P3SV`l?*JQ~V!^0F0|40723i_k&QBBFdY$z*PZ`t5'
    b'MD?Ysnhx>3S2f4+FOF4Zlbw8sn<M~J<#<%JigfIg)F}#R$f=Onui+&cQNHv`D;rl}HEyANq3cN7=0b0}eJM$&Dg-=?cJ_)24<+OI'
    b'&)YqjB0L$QT^O_$OhI|fac2$*&De*$sDm)avh=R6M$pAq@?p#4_^1M{uE4~wB?Ddz%DbOYXII37GXK`SgbOC49w~I%Wi+LW87^8&'
    b'j7UWY&F)X3C-((Q&~lwSUula&o{kX-K^e6nW>nI3Mnm=!JMuS&CgRU>d+rcTH5ySHVyMTdE9J-A*_T$FW!J7Xab(kV8uB#c)^nB@'
    b'YpE{GZ`Er6y`oz?S@Txe7YFIE(S;xJ)sI|WyLg~_2q}(6GX=GC7w_XWMYfoehga;s0nJ1Q3<FVJx@Wrd(<kX>!g&R$5~Cc%PcUwD'
    b')<D~4_F}AYi}yv(=R^g<U`8JI>fW^jC9$Nak-6zm#l*FgE5#l&rv_w0vYrv^e4&>N{#l?ltYXqUwsl1Qp-~>n#E17aJ;~*v9*hsf'
    b'tupxj)kw3_R|%kc@~KO+eK#sNP7zrfDG`KJQRXT4gE<o2{dM0Lx8wI7sRTOw_UGT8B#-|P0@QfwOU|`OLkl&m13();v{G(WT-*k3'
    b';xpG9!G(UpTA=dafnZbLs>xp)ZnuG;={$XQ&e?dlDt!rBxdL$G8(KK!K;stu>~YMTXE`cX7L?-Br@u)<D##?q(-xo|2chq*GnrV6'
    b'_eyz<{O;G}LyTtICVpF_83dUs#d2{2-up4kaRPMUhubdXy2f9whF}#Xm!T9D%$b#m#xo8clcw|3;;f{XAi{?Xvg@*|3hYuM*S9NY'
    b'tGl$NEsad_*T_aGwu$~f`PYTlf&#!D)rBhoZRQ?-b!g>@;4H;luvoa%_m#BIeXykb4_G!W)03Ajithn407!mkanxw13STrPDt!z?'
    b'i}jD+C|X+9GE(%ipNy?|cKr#@kDn;fw<DNT_v}$}+fOmv!rxK3`gPJJx&aS&^%(pt3EAoSlplP0e7?=PFF$e2^hkFUmIQ7a4<bvk'
    b'yD}VR*zYC{ZHMrS*{{ea4|U<OJr}qAR~2lzQ{DR75SPEEPSFGI7eaiz7gGBjY&@HOWCW!rAZuG6A7eTbL=b*K#Malm2|hvlLt6ld'
    b'a^PEwIQv-Oh{Kr}!~QU=G?j#u6SK#6tbHm>O#@VAS^AZDkXu~~pL34#8gQfy>ed`+G?kW5Vfho;f%Gu;T!15}64~Mxs9`~gt~Yx@'
    b'M=iKFaSdmbgU+}#FNsmyJ{-begV+91TimOIX2P5HA?j7o#x&vLbm~3Bz8Q8Rzv~XP_Vk!vrK}~=vJ<d&AAuf6UHWr_a-QxeF&5cW'
    b'ppD-ANBwFdp`r_^W~p0AdeY-z%=EPL1<DhQxt5C@J^VA*hbhrdzX2=p9}h!0T>y`x0M($1@!Lf3e;5L|6}W?%;I-Uaglc4HVlBP}'
    b'sO55?CtigZ!z2?ml_g(pWEsmPWcuoUf{f<|$NZ#+qCfY*cb&%}be?Z%MOuak$YXG8SOqLFj8criVQs@dYD~&V9gv`Wn|=O$4B;tW'
    b'!A;<Wu+1_+{e3U9tjmlUKRSHdJ|;*0CINLjuGkdb;q|(CJLvDIxZmR%z9$t_)DBOmbRd{aoCoOxO8o=P(RM}mdSw(oOlPVu0{q@x'
    b'3Sdyh0z?%*#E{*f&9@tYtYeOOo_;V#jwT(F$vjL&OEB;N1!$9k@}DA{KuR4r_EKn>j@%%PEo9s~<<SF6q8f@!2+SOh4UA(sfaASj'
    b';|%okAmsFMVqiVC=qgA6v2hi+0uOOY6@Dl5={rFgm#`li4PH>1^6Q7mlKi<6455S4FQZr7yY@d`)s(Q2QO{WS|L$ZN(BbXrM{TJT'
    b'^*{*d@vgzh`gf;CCMiJhmPrdX;y^{VismS?uRCG5-_)DXzy?Kr&ZF_KEsw71U+B%`kn2fjxHz5!>GTUT1sytkk1;w}XZLMzVdH_s'
    b'MOl#BfG5p0$k#x4@wPU%tUTg^%&k~ll^R7r$$LhN;9TE-k?@B)v&55)GHRJCn4R<NyU5qTeuLHIQ^M^H<ck{^_hNjvV`>fOyt=S~'
    b'M3_i-1XhDtKGHWBc${E2BsLH)K)zXrehUzrxaoJ5v{WDS6a-M9s5e}@)T6E~t7Ug;S8L_)R9MYeH5iDdR0u_#?a$TH?&)#XN{{2u'
    b'8~)zGixf?ydNa(8ULPtN;v9-ZH#$pmneZMMQE~2GfQ$PRK7SH2R2M<VG3!?^weu}iaiQD#0rx*C%XOH)$Oe|d_Z{0`@5CBsS<u0-'
    b'o|7*HCus4<{Qd&VaMt<`Lf}B?81KJeZI-nkPkK_;1Ox0f?cb_T3b2xBOzur78~jDy@=Y$v<Pb^RiNyKZKjOu$=3^~2U=~L5J;7Ka'
    b'A$G8hLPz^WB$JXBardDE&Li%w2K=js^@@;XF~oo6AW2^f<6KQIJBRKMQfBlcIl2@;L07fyN&7*Bb}T+Yzsvg~`1R;HG%90w%26xf'
    b'5O!cPzo0Fya*dFcU|=q87rpc1G}9*X9tgb3Hh}|FGDCeXsizyxZnC=6kQGM7sjm6%xOe(-919=EO=q;RaxqLfyw*rQIB~`%Ckwk&'
    b'=aI)yFRY1Icgb|qNvpW@%UPFFjtx~EUsk7^d@_UpC%$YCr^7%+<sqssT=f9arq>}TcFh2gE-R-aGrL=FjKrvLM8G^Mgt4!k5S0b`'
    b'UUK)m(Ylos(IH_GYri+vn-ap#qFxF9z8(2^_gQX~%zV`?FjYPha1LP5<UlZO6kloS?rHi8J%C+8P54Xico#aaff9&4Q@04{WZlau'
    b'IGY)h_V(K$C(=xPGq8F{+1-56muxx_HXg&$`PJoa9occl0I8_hRq1nac!aZcJZPhzg!gx0WL1$P$fBgvD%qL)UJf60g0%%u@>=+6'
    b'4-$@Sz$V?wNj5a#UO=V0{a{{%=a`pD<N}aHM<9`lL2DO%lb2INZoJ;d4$8<yiGm=eB^YPB5Fs_Hft1UQgqnJRg;O=OTpqyfi79of'
    b'lvN?GD%bWqQuH#(k3BM1XxO(tQH8DX&FH5m3MMFU+i|d5+~q^n7x8v~ssIf?9*Hq9cBqS2xfmAPfX>4x1-Xr}3jPk`-jGpGRe1|}'
    b'j(0<e_$X(76Fv-$gju(~=~wUK#k0Ke76{rff3YUn>EyWQ+1rtL$MxTMoL<>CB{L~_utIx*EMilp-|B`#tsl7TXS}9SK8e?4bxqwj'
    b'h9rcxWYT%y2jWui=YXfLveQa^Veo)ufIyFF9@$5lLTplKCLs?Q(UDU=<WknPM6kf&T==K-0$g=vLc@tII!%!-=&ZkCTO>m6bMY{9'
    b'+t4p!I=tT)b)T>1MvT0Lx>P04N6#LQ63>w1VC%sJLhU-UuY5|FE{mP*s~{@`M`q6MR+&_rwjm5M{fUC_zQ-9GmG6^j_K`&+9)3<G'
    b'Rp_D~$=Z-jY*-I>DXeI!tx`iMuxD@Z3nO-e?Dso?N*|Qfh7f^puGX>KT(-d2tlC~WCDha;<7e@sRbb^RPHh=+6v(OZN)hXRshn&&'
    b'1e-Ce{KwQW8u$rNAyo?z;q$@x7BL;~cu~Jq*}ljsRc5F+>yF|9l6hH_@=FfWH}|-K3pwZ=@cQS2eTOkZ>KXsi2gDe#0}~e8b>f^F'
    b'&NaECKCZ0xq{`c8lZY{dsBH~>PFsY}Ry@23N=lh`BiI&UJW7oyu0=WkSX$_$etj2>ZP$Gp2iKGXc$^rt)LZ|)WXa~t5O;aTQ_sBu'
    b'?;9Bw+kA(TqAhm&&kV{PKPRECK3I$bs!{dDJ*n5ensG4oJnx#wsy6>Ilv3fi;{_nvz!-N9t)=W`%^&-eo$}_vD$l9Ny_#e=?!2TF'
    b'dkr0P-2bsYW)?3fnS7o4DL9kNRGOiA-XN2=e=MRyG3g+~{KnYqZeYYK)N@->&?QfSrR9A04Ht;k&co%@(eGx`F56nDr$G!BJiUhs'
    b'9$<|ijafEA&hEEZ^*VoPTst$3UEpj&W{H$ibr(FRJ=KUSB(_M*n&tVxSlrQ6G05U?S0Mm57q3L6P(TFTUc<?gKiwgKSW_8uLWg1V'
    b'|0RMXMxpG*+OaZMQLtQLGimdC@J`A+9GZAT2X0zBYG3KPSRXAZ*X%^Jr^3nhmiooG#1iK@%RLzQk&Sh(uPsSc%s5Bh*!V^Q57&b>'
    b'XV9U0ea;;?YSe~pP#CHu24*;dCh#cd5YcP0Ma?b4mUIG$bMwyQv18j_sXJx30i}D6{<z9N6dvWbfnVTs&}#)Ly<m;wRg+Whi`O!('
    b'K&f9O3Lz_Jy=HxiHu7`QYIS=df0DY@l<PF!YOX3oO<d2yhuDj?Wlca>i*1F4VML{H&#s{JZ0`~1^gY~d-X)wBt$t2>0btLK>Y0VG'
    b'0F#)L@OpdNEju-RAGzVNk@e6_aYwJ0;o{Rz0&ywtJOJWws{|*Uu3n+G)J>(PAP3SwUiSGHj26<+8d>Y4K!x`2Wo9L-K^a909Nzx<'
    b'ovAe?k~J2VG-wY`NP}doI~>(tFQZQ_DZ$HQ5Fr6L8w3jNmI&w-z`7JvHz_{}vW1z^ZQUW6?n+4W63O(BuBX-#IgIf}4&qK1EcQd}'
    b'3=498{a`T&w}m*R`2jkjtsyNTb-c`5u~gt^Zhs$RuIB)=LJYKt1#7vMs^SS9I%6>eYzWd_HmEv}q<sM9#{)<v$lmB0K&&O<p0PPf'
    b'S4J?Ks7lv@i<>8SyHnQ6!E&WMBo}C6nnMm{07gv1ADAcV?UT2bN&#@~hZg$VN~!8hjjoTfn|x(nHl&1od0`U3O;TV%5~YUQ6ugb('
    b'fNsW^^CTVB>2|Ed;Cr#sV@nLN<4Kp4zd#$4L01#IW~3wu@tI{gj<qL~OcZf;zq+Z^<-}gi;kW*+aa1?mQaTCufRWDVIY}_KKFsOk'
    b'8Hq1^?8<^kAEXs^d!<8*XJ{-J1=+)6c~8Z4|E|55x%z?|Rwn;^`kIp5c8cxnN$9IK2gj9oIp$Aj_A}8jlre&4Gic7;?v$U$@y;8='
    b';vD9*Ls>;5IL&3PV(6+&%OXW~?bTv`a^K_O8>KxR!O?&-UCeb@yr4&db$pHhc>EF-?vq;qc5iwpqno}E$Ym5Qu;+EWb;Zkj*cDF|'
    b'6FXnqm9Qt-T!L5E(!f-ZNq9p|D9yw!HN0pwIudqkG|4%XwIvzKF-wQov3Wc$s0xvb_`7g84WgI&Z`0Y8xlw&%{|FnsN@N|(Dh}FX'
    b'E}lpXJn{|B=cV_WRLFlPW~^nM-(-@3clb?6{CEu8+|_c^9+>5yd@=RKcm!`TitO!9`3mIr_YCTJuHopHsVgRY43%9UxL*cH|IA10'
    b'yQG~<s%=t*wSZEu)uHR(kDFT;n$EbAqstpR9F`dl8ZSzQvwfPzKSUEBgxXW=5t$6-aY_`Hg}!t4-&Ne9q95GtI;eTUWCSUJdo+qf'
    b'KH!q<@7(koI-zIp6X%L<)q%N(nc0&E8a-|&!<Z5Xtqc?G@3Qb??3Tt#DM*+dLa>x)avgtbUOtJh3e&nl!l)kgy@RE-cUetLGHko('
    b'`erY`Ie*eGmhq65>@@k}BJg5@I}^!+EBdg|K=(0^p@$1h7^uioYaJ%c25fdm;=nNjQ&qZzj61oAG?EzT%=ZE$hD1s%4c1#pVfn%E'
    b'JG6oz{aKQ0{fr7qc&^lf%}o-(llVzLxpGgi!DnZKfsA0h!h{=ybjMPc35Z-d>UG-kyHAkgack+li3&sO6Gf3SJWY6`!#Sh+3kHhD'
    b'JZf#DRhbW{xS*TlfmkXg)N(d@uGFVNItW}!(2RTYR`bM(#yMVL>~|3?6?e-Bs>;CjvJ(YY4PboW+N#ED70{;p^i-RhFbuXSJ_4WP'
    b'z|)~3{VMbsavmcC$))lFZJX0CKF<-=gyDy}%5+v~m&gh#q7WSl5<ZX6tn|<y7w9j_$oRyhR;thvJV}#5XKnq8C0@bdz*sIKCpC{H'
    b'ewi$0Y)O{Nzc@Q}+$rDg4tNb%#2e$c4F^5;<Fq?n<ltsp&5#^89D?_l6wv#uMiH5Cp;_h*<ZOQx>!6<Zu+^3yePR)CM7h$dPIhCe'
    b'3Au-ilH3X_{jo=(jMxi5+$huNM(`cK(y84#=Cy!*s0L4e_x_Z>_a9`&|0gEmu1|^Ypa1PKE@o(y^8Ba?5@uA{;I0yYP}N6p?D!u_'
    b'{@b9$CI-6Q;c#w#jWv25O$HQ9lbNlA4~x6EQpJEyO1>}zG5m7f*7><+BN4(>WmP|$uetngej!Zdy$YRnVpMlCrAPVn8*t8!npJ$q'
    b')DwvI9=0uw5TAef9P7at#<;9A_+B<bV&NT7Wj|tf_N3y-=e;hZqSW9=&D8DtyiYk`*N__bYc;~a+XDEO^)=ul_qyo6$|O5{Ff+g#'
    b'AeeWl^OG?($pBk=3Uv|4w<qiSZuqGnrS%)u6`)unp{<_W0VdRSau+!k6Zyw?Ee(`fkU5tWH9X=G1^o$K>A>JjAw3#&Aho*vhosEI'
    b'^WJSVA~M>os?n52)H@`&|82a4N2@`;K4$F7b$r&?7q!vqh;72=Ef6aWW=BP?;>=y<<miJFik!_dhi9+}Oh5_R7+`t4F=#bzkZ7|U'
    b';i6q|8`B?zDpsIFT^M-V;ip6wUL@VzY+BT>|0fWale;CiTZVU)4fA0u%V&ewdLV<?Y<gY1wd2S*gtoL+EN=}t0iOTjEW5i9)?7pj'
    b'&N4fMh+zJ+NTsAbC~)htMJ`b|n=ezlc?}$Ycuta+JdO0uYObT+s+8X$FRa}Gcr$!}i2}sywVb3d9F3x?dK7bMR9{`TqFvqF%Sfgp'
    b'5D!-nc3CG-PG~su&Bt}q6B}T!FDnj|<+HL2lH10Fv7&9$*|iOBE#HI3lB458+B3hO5fLO;OROgw<FpN0s}1ZXV+n~|Y_ko7kkFQe'
    b'XK^A-c))=gnwY)QXLlPGX`tO6n+;f2AfE(U1?Mo<1VHjE<WS>=lOZI81YM>#`d4WjY$wQUHOopw{vxbmA%fD}qBh_@5cPp{Sy11e'
    b'|NSwP6kQ{J?iT<B(U@;&4ZzX|E0zUxX?v-Ky2RGtX~DiDYSjBsNx3ysFtmh_(B>v8C0s<7H3WINVk=E629QAfYJhcGrUXahw=X*n'
    b'2Ri)=Pg`AY%E{`?@jWKOpMlt|g7IY`cyN{`0y^(XwImy$-ty6DU&|b7C;zcX-W03S#0XCYV&0Pw^_cPVd1fa{*_gJqII)LfjnN{J'
    b'cUgSHgjYe-gw_YMpvv8rz4~`SoDBgyLiJNyo}=sjr~8Zp%6AJyAwmuubkzJn!W1X3@@Npf7x|z*3O7$Xb=Sx2rCld_Ox3h;@p3M}'
    b'b2V-XXXk~Z&R4(;lW$L5K3l+TPK4^{eYfI+H<-fI8#u5V_FL1zSC*c5vOb)oz5{xgPBLEK4XwaPiye^ow#}J{@|lTKCE?ccOANl5'
    b'bL?24<0MJ(5d%x?@9DVoeVv<Dy_c$R)mD>_5q1)^ZdjyGiOHAhKBZ1xBbATF;`mir$4dEe_v}p7J+A-yMyDQFFB$em$S{Mc8H6Nt'
    b'Z%+n+5gj60l`bQO8F57%?=SPJ*;kz3m34WHA8xAQ;%c?#<CIP8k3Gt{3wXjl9G^nY2-+tEw>iYvaY*6kPHqH$<(@*UOq^nduzZ0)'
    b'@E?Gpo4u^YR0g;7l7i}7-vOX)tNBPi5F%#xuNUy;ULe)~d*!Ffn22?6Po}&aJE}4)YVB+F9Aw!-HroLFo7~CE`|FTu{P`Kx6Tlq~'
    b'G|%D+#GlubyE1s+x!gre{(&vo>U42j$n7&wkVr;yh^w=(>u@X8L&@0wZQ|hezhnzvbdP5zaqscV;}T!*%}P-z-4@eOfhVwS)lqRc'
    b'uC*wR?h5V>7G+QKshDAdUj*OHMJ@XK!^7?vwrw#FLl7?>-U(&3Fkwy|6x|8^>^-Fori4h{OEwvRdPjg90m%jv6?pl1Sz5lr-QtL&'
    b'3XhTy@WwjkGt`V*+IH46HQ^J6!k;-vb(oDFwS0h^M8KkxT|skMj@2H{e&RbmscPfXSoYZfUganRBkqT5#sCWuIs5cZ_Nu9C0?yJK'
    b'Cpm$#KcowPEp3ma)=*Pj8sZeVl>`MKTH;I%x}4{Y{DBx&i`@Qx;tjTL-h$xDsJK?)&Dhz`{y@HEqgSbVvLbppGF08gR1SV~_1AME'
    b'@8Q3;8?uWDrz{2`G3_a-RDl5K{YO1v2JrQ$9PR0QGOjQ$;gJsDs?l%-HaAzZC{31{K`MN+G0FICyM*71ljkxtdi!(3(5>GPPVDGU'
    b'3;67XKRxA;*8ezx`=0<t-CSlpkYsBtIkerHcdn#ZXk)BAodAlEnYYyoNSD1uk99YigcVb?mucW+tda^__j1g@nKVQXR7yjW#4P(q'
    b'P-d(rVL4y1hX^k9Ak-(h*U~;RD0$A4z$0fsf;_$@%ec*p&cbFK<{Fb^Qgx+l=N~ICeuV-s*)T?vde$Ntl})s<mMVk8WacJ@UM7$o'
    b'mRW8D%HboCRpdN+)AKjI9Z+2@fqXIm8w_8XJf@95>~8a>d9#_Ao16w0pCyuWd{FgO=DX5u5rBeYwESWFdqy*!4M8W80odw4^aa|7'
    b'NpjDvYhWc-y~kW6g9JFqoi|PWDaRz1;1uiKs+G`j-7{oRYvfpVC#Oir#;36(bUBFp%cNsE`J(&VP;;tZoL17Y(e6q>I5*w*Q`HAW'
    b'qp%0DbXhUQg|<PlT4mCr(hCbz3^KLy8l|)%+Aj~8wbM@xRLU^G3_>!QI-)*e(a_4`i1`5b?i0F%j?B?bcFMYp47YeiWH7(lF@cxp'
    b'4arTv-=0!ep%Kr&e^%lpjox3jhFL<o$eLT0fWlSP%bqcVOltZ<iNKHExQ1Vllc}{+Vt#hZ?~~?cQf!Et?riJIP)&d8>psYQI&7Q7'
    b'RlsdNhn6Xzu8tLj0D9W>_Cz;!kwz&5q<EQ{+oqG4U%-rxX4OFsEU6KjGLp=gRG^Og`Qh)*Hqohr$5KMt|8{PP8K9zOxUne&Wt*tx'
    b'6Se5Vu0PJGJb69<m|s2d!BtON*cJZ(^z`Jrd2PN&#O#`6u+4;Q(;+#Kup|W=%8gm?(dZ~7jvTP*+q)3OTlnG4LRJi$R`w#`dnt2%'
    b'YO5el4}RB((XUQ}OcR^ES11Q@;b(9-gyORqAc3;kP)W3P-=+|!zc1}ZVi(g9{(^^qIMg5?lFV#-S1l06Mfnpi)$)gy*NE%uej=mL'
    b'm5e`^swgVSQuZUeT#ljXdo;TB-dj?&!`u$*Z#+G$4Bb}_p9a{QEb4g(xEJJwI(|Fa9E>83o$b7A$v%Vq^{_s~4}kB)n~-}dJd~}Q'
    b'z9#L9=<1|Ao7;OaN|`4)ne~!PeZJGR*g^S8e<9R~2QCCI@o`(@tjpfd2*pg$9Lr2)08yu~nT;}9OxBke^DNJ&^(496EXb`eqZD|5'
    b'{jtgW#~>KO2juT=Ym>R_{LqQ-^(Yc+W74*<HZuKXXZz>Ye_!ASMKK}xSwsqgoH2+JKGc7RW55L4Ca6s}XVPUM+WSmD=T`x0<gBlw'
    b'$iJ}cEwP5}Px*S5Oo^u!93f<sT0yDWM)^!D41G0HXgiPIl#(${&nNRj@oC3Z!hrP-2G!LRtzK4_$LzX~#eAm6@tTNB)vIXfyzB~7'
    b'6!BBiG<*_XL%omzKbcPzSCg4%j_!+=@x_NiVH`}EzAJ55fD~h4Y)(#l=N}!~#<OLUYeFVNY<{mOSc>@!u*h+yE$nj2zAN$UwOQ!S'
    b'cpK#~J0oNz?He1>)T+2dETnH$HeLgq^H1d&oR&ON?N$l4Vl9u#6v7I#&m9f@@lTDI?!GA_KmYDV&Q^udt~4WO=wp1|+-f#ad4v*{'
    b'0tl#4oz@&FaX*$sTM~9_C~nj24;4$nd61xwUwiXv*HARwvS31GYbpv??`*INCsF2jONYjYh=F(4W5R0`2gf%8W;X=~5=($$-AN%c'
    b'y8IW1b<4cIGBv!5E|}Pt;@L>*!ru+Cl(BJXZc}xIO#2IVEg=^OIesD)3Fi7^iiJ`7rzg(SE75rA6d%qy%gio#Tdi~2tO+O*MrKF>'
    b'CP8cq#B$oe>f51yR9iFeyj#^@Y$F4=0>xBiS9n@-+Ho39YHBX?pIIbSuY**D@h@BEv&Y#)GjTb$%rl@0^WUm8w6lZV`r>1^_c}~z'
    b'@g~Ww@At%K3<i^UOktGc&YfcWKYq&<g{S6KsZYV6{!_G1loM?*-QmCElWRk;I~gONJT49En5SKLGa7~m3mE;@gZ2Yo63-P8y{4-k'
    b'>V(ru)qYddo}F)2yDavX0#`$FkNFhCMBJDA-x3fIbxc72-`p{W>I3F?k5L?Q&kY&`P_DJYb5UbsCI6IZMyo89pI^bIaap3fpv`v~'
    b'X(n}7z||mCU%XK0qRFVf)kSoL(u^FF%P|uxC<v#TKvD{pwwb(#3K|U+>gW51q++Mx3EVZSs8eWd4elyopYMio*W&hy%LoyGWNF~o'
    b'6~gt+0S+#&WF4^aZ#A_NR#q&k2&|jjuQ1;kZ>r!kHIlAL*HH6<&xY>=HgRy(dTKYY1gCc*>BvJg!(*~t=6zNu()T-9ELVQn%BPPa'
    b'_7?c8xy7QO!fO52Dx&q02J@|TXCBtB+##=3^JE$Cp)uC<PpS&oz1U5x8&30Kvylg5Y?>DdB31h=sT|O$3}{@$Q2s(V+{g>yDo}hp'
    b'>UtaZj%ipKhqN4<Q{i}MUwQ2Vu38O<Y1;&bj-i?SCjU4PTtC>RY$5OQAXklhkDAz{EbhH76abe8#p~_}u!YKyTk=>uUExl4Z<%m}'
    b'A}+D)11iyOSVWMNr-V}D!JoMX00000i_));X9ITb00HrP#hwHJb%E<avBYQl0ssI200dcD'
)))
    for si,di,core_recipe,sa,sb,br,ai,bi,is_reason in records:
        source,donor=json.loads(lines[si]),json.loads(lines[di])
        refs=(source['prompt'],source['response'])
        core=restore(core_recipe,refs)
        other=restore(br,refs+(core,))
        for slot,original,suffix,answer,n,reason in [(si,source,sa,source['response'],ai,is_reason),(di,donor,sb,other,bi,not is_reason)]:
            row=dict(source)
            row.update(id=original['id'],prompt=core+suffixes[suffix],response=answer,input_tokens=n,
                       supervised_tokens=original['supervised_tokens'],total_tokens=n+original['supervised_tokens'],
                       baseline_id=original['id'],source_id=source['id'],pair_mode='reason' if reason else 'short')
            lines[slot]=(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8')
    data=b''.join(lines)
    got=hashlib.sha256(data).hexdigest()
    if got!=TARGET:raise ValueError('reconstructed SHA-256 mismatch: '+got)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    part=None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent,prefix='.paired-incomplete-',delete=False) as f:
            part=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
        # Publish with a same-filesystem hard link, avoiding overwrites even on a concurrent retry.
        try: os.link(part,args.output)
        except FileExistsError:
            if hashlib.sha256(args.output.read_bytes()).hexdigest()!=TARGET:raise
    finally:
        if part is not None:part.unlink(missing_ok=True)
    print(got,args.output,len(lines),'rows',len(data),'bytes')

if __name__=='__main__':main()

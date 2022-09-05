/*
OpenCL RandomForestClassifier
classifier_class_name = ObjectSegmenter
feature_specification = gaussian_blur=1 difference_of_gaussian=1 laplace_box_of_gaussian_blur=1 sobel_of_gaussian_blur=1
num_ground_truth_dimensions = 2
num_classes = 2
num_features = 4
max_depth = 2
num_trees = 100
feature_importances = 0.6255651136322791,0.01342646095592343,0.047769471490238476,0.313238953921559
positive_class_identifier = 2
apoc_version = 0.8.1
*/
__kernel void predict (IMAGE_in0_TYPE in0, IMAGE_in1_TYPE in1, IMAGE_in2_TYPE in2, IMAGE_in3_TYPE in3, IMAGE_out_TYPE out) {
 sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
 const int x = get_global_id(0);
 const int y = get_global_id(1);
 const int z = get_global_id(2);
 float i0 = READ_IMAGE(in0, sampler, POS_in0_INSTANCE(x,y,z,0)).x;
 float i1 = READ_IMAGE(in1, sampler, POS_in1_INSTANCE(x,y,z,0)).x;
 float i2 = READ_IMAGE(in2, sampler, POS_in2_INSTANCE(x,y,z,0)).x;
 float i3 = READ_IMAGE(in3, sampler, POS_in3_INSTANCE(x,y,z,0)).x;
 float s0=0;
 float s1=0;
if(i0<10.925628662109375){
 if(i2<-2.915651798248291){
  s0+=688.0;
  s1+=221.0;
 } else {
  s0+=3672.0;
  s1+=316.0;
 }
} else {
 if(i3<40.5782585144043){
  s0+=652.0;
  s1+=2301.0;
 } else {
  s0+=253.0;
  s1+=212.0;
 }
}
if(i0<7.047927379608154){
 if(i0<4.23972749710083){
  s0+=3041.0;
  s1+=54.0;
 } else {
  s0+=716.0;
  s1+=167.0;
 }
} else {
 if(i3<39.927345275878906){
  s0+=1189.0;
  s1+=2683.0;
 } else {
  s0+=251.0;
  s1+=214.0;
 }
}
if(i0<6.8637590408325195){
 if(i0<4.811124801635742){
  s0+=3293.0;
  s1+=55.0;
 } else {
  s0+=547.0;
  s1+=146.0;
 }
} else {
 if(i3<40.301963806152344){
  s0+=1168.0;
  s1+=2692.0;
 } else {
  s0+=212.0;
  s1+=202.0;
 }
}
if(i0<6.489502906799316){
 if(i0<5.1971282958984375){
  s0+=3349.0;
  s1+=86.0;
 } else {
  s0+=308.0;
  s1+=101.0;
 }
} else {
 if(i0<12.779060363769531){
  s0+=745.0;
  s1+=584.0;
 } else {
  s0+=735.0;
  s1+=2407.0;
 }
}
if(i0<6.816415786743164){
 if(i3<10.207512855529785){
  s0+=3177.0;
  s1+=83.0;
 } else {
  s0+=533.0;
  s1+=140.0;
 }
} else {
 if(i2<6.188753128051758){
  s0+=1241.0;
  s1+=2291.0;
 } else {
  s0+=224.0;
  s1+=626.0;
 }
}
if(i3<11.191299438476562){
 if(i3<6.831116676330566){
  s0+=2815.0;
  s1+=153.0;
 } else {
  s0+=796.0;
  s1+=293.0;
 }
} else {
 if(i0<11.996983528137207){
  s0+=837.0;
  s1+=462.0;
 } else {
  s0+=769.0;
  s1+=2190.0;
 }
}
if(i3<11.61424446105957){
 if(i0<16.835834503173828){
  s0+=3575.0;
  s1+=219.0;
 } else {
  s0+=46.0;
  s1+=258.0;
 }
} else {
 if(i0<11.408758163452148){
  s0+=748.0;
  s1+=439.0;
 } else {
  s0+=820.0;
  s1+=2210.0;
 }
}
if(i0<11.416803359985352){
 if(i1<-0.30033349990844727){
  s0+=456.0;
  s1+=189.0;
 } else {
  s0+=3842.0;
  s1+=376.0;
 }
} else {
 if(i0<43.133270263671875){
  s0+=772.0;
  s1+=2510.0;
 } else {
  s0+=109.0;
  s1+=61.0;
 }
}
if(i3<10.966110229492188){
 if(i0<20.36196517944336){
  s0+=3571.0;
  s1+=186.0;
 } else {
  s0+=27.0;
  s1+=201.0;
 }
} else {
 if(i0<13.169004440307617){
  s0+=965.0;
  s1+=552.0;
 } else {
  s0+=713.0;
  s1+=2100.0;
 }
}
if(i0<8.728116989135742){
 if(i2<-2.9277777671813965){
  s0+=593.0;
  s1+=154.0;
 } else {
  s0+=3412.0;
  s1+=210.0;
 }
} else {
 if(i0<13.457650184631348){
  s0+=469.0;
  s1+=432.0;
 } else {
  s0+=746.0;
  s1+=2299.0;
 }
}
if(i0<6.226564407348633){
 if(i0<4.522761344909668){
  s0+=3162.0;
  s1+=52.0;
 } else {
  s0+=462.0;
  s1+=110.0;
 }
} else {
 if(i0<14.472342491149902){
  s0+=859.0;
  s1+=748.0;
 } else {
  s0+=687.0;
  s1+=2235.0;
 }
}
if(i2<4.537585258483887){
 if(i3<12.410600662231445){
  s0+=3648.0;
  s1+=415.0;
 } else {
  s0+=1207.0;
  s1+=1929.0;
 }
} else {
 if(i2<10.424745559692383){
  s0+=241.0;
  s1+=506.0;
 } else {
  s0+=74.0;
  s1+=295.0;
 }
}
if(i0<7.085631370544434){
 if(i0<4.8004913330078125){
  s0+=3157.0;
  s1+=73.0;
 } else {
  s0+=594.0;
  s1+=149.0;
 }
} else {
 if(i3<39.75975799560547){
  s0+=1191.0;
  s1+=2692.0;
 } else {
  s0+=255.0;
  s1+=204.0;
 }
}
if(i0<10.180667877197266){
 if(i3<12.446346282958984){
  s0+=3708.0;
  s1+=191.0;
 } else {
  s0+=580.0;
  s1+=293.0;
 }
} else {
 if(i0<20.18325424194336){
  s0+=582.0;
  s1+=985.0;
 } else {
  s0+=401.0;
  s1+=1575.0;
 }
}
if(i0<11.306407928466797){
 if(i3<12.37750244140625){
  s0+=3609.0;
  s1+=202.0;
 } else {
  s0+=687.0;
  s1+=357.0;
 }
} else {
 if(i3<40.1893424987793){
  s0+=624.0;
  s1+=2349.0;
 } else {
  s0+=277.0;
  s1+=210.0;
 }
}
if(i0<6.485222339630127){
 if(i2<-9.765020370483398){
  s0+=1.0;
  s1+=14.0;
 } else {
  s0+=3699.0;
  s1+=183.0;
 }
} else {
 if(i1<-1.3530874252319336){
  s0+=16.0;
  s1+=7.0;
 } else {
  s0+=1455.0;
  s1+=2940.0;
 }
}
if(i3<12.443806648254395){
 if(i0<18.293594360351562){
  s0+=3660.0;
  s1+=279.0;
 } else {
  s0+=35.0;
  s1+=285.0;
 }
} else {
 if(i1<0.442018985748291){
  s0+=1251.0;
  s1+=1977.0;
 } else {
  s0+=223.0;
  s1+=605.0;
 }
}
if(i0<6.580826759338379){
 if(i3<10.107500076293945){
  s0+=3193.0;
  s1+=58.0;
 } else {
  s0+=497.0;
  s1+=89.0;
 }
} else {
 if(i2<-6.260214805603027){
  s0+=262.0;
  s1+=324.0;
 } else {
  s0+=1260.0;
  s1+=2632.0;
 }
}
if(i0<7.05680513381958){
 if(i2<-2.733414649963379){
  s0+=600.0;
  s1+=102.0;
 } else {
  s0+=3125.0;
  s1+=118.0;
 }
} else {
 if(i0<12.366382598876953){
  s0+=586.0;
  s1+=475.0;
 } else {
  s0+=782.0;
  s1+=2527.0;
 }
}
if(i0<7.078171253204346){
 if(i2<-2.915651798248291){
  s0+=542.0;
  s1+=102.0;
 } else {
  s0+=3209.0;
  s1+=112.0;
 }
} else {
 if(i0<12.77786636352539){
  s0+=661.0;
  s1+=517.0;
 } else {
  s0+=778.0;
  s1+=2394.0;
 }
}
if(i3<12.428855895996094){
 if(i0<10.431231498718262){
  s0+=3656.0;
  s1+=211.0;
 } else {
  s0+=119.0;
  s1+=360.0;
 }
} else {
 if(i3<45.79316711425781){
  s0+=1194.0;
  s1+=2460.0;
 } else {
  s0+=183.0;
  s1+=132.0;
 }
}
if(i2<4.392390251159668){
 if(i3<10.886513710021973){
  s0+=3471.0;
  s1+=309.0;
 } else {
  s0+=1391.0;
  s1+=1950.0;
 }
} else {
 if(i0<15.235268592834473){
  s0+=165.0;
  s1+=59.0;
 } else {
  s0+=229.0;
  s1+=741.0;
 }
}
if(i2<5.135528087615967){
 if(i0<6.416987419128418){
  s0+=3700.0;
  s1+=174.0;
 } else {
  s0+=1213.0;
  s1+=2155.0;
 }
} else {
 if(i3<6.273715972900391){
  s0+=46.0;
  s1+=21.0;
 } else {
  s0+=281.0;
  s1+=725.0;
 }
}
if(i0<6.8637590408325195){
 if(i3<10.207512855529785){
  s0+=3167.0;
  s1+=63.0;
 } else {
  s0+=523.0;
  s1+=127.0;
 }
} else {
 if(i0<12.125296592712402){
  s0+=633.0;
  s1+=509.0;
 } else {
  s0+=831.0;
  s1+=2462.0;
 }
}
if(i3<11.138038635253906){
 if(i2<4.893100261688232){
  s0+=3502.0;
  s1+=382.0;
 } else {
  s0+=66.0;
  s1+=89.0;
 }
} else {
 if(i0<12.001962661743164){
  s0+=852.0;
  s1+=467.0;
 } else {
  s0+=765.0;
  s1+=2192.0;
 }
}
if(i2<4.392390251159668){
 if(i3<10.885807037353516){
  s0+=3406.0;
  s1+=323.0;
 } else {
  s0+=1355.0;
  s1+=2001.0;
 }
} else {
 if(i3<6.273715972900391){
  s0+=45.0;
  s1+=23.0;
 } else {
  s0+=353.0;
  s1+=809.0;
 }
}
if(i3<10.478763580322266){
 if(i3<6.623052597045898){
  s0+=2801.0;
  s1+=159.0;
 } else {
  s0+=722.0;
  s1+=250.0;
 }
} else {
 if(i3<16.433902740478516){
  s0+=639.0;
  s1+=568.0;
 } else {
  s0+=1087.0;
  s1+=2089.0;
 }
}
if(i0<10.213205337524414){
 if(i3<11.961465835571289){
  s0+=3544.0;
  s1+=185.0;
 } else {
  s0+=632.0;
  s1+=327.0;
 }
} else {
 if(i0<13.069133758544922){
  s0+=247.0;
  s1+=259.0;
 } else {
  s0+=713.0;
  s1+=2408.0;
 }
}
if(i3<10.688507080078125){
 if(i3<6.719602108001709){
  s0+=2694.0;
  s1+=151.0;
 } else {
  s0+=708.0;
  s1+=289.0;
 }
} else {
 if(i3<15.599727630615234){
  s0+=548.0;
  s1+=490.0;
 } else {
  s0+=1203.0;
  s1+=2232.0;
 }
}
if(i2<3.3714816570281982){
 if(i1<-0.2972067594528198){
  s0+=724.0;
  s1+=863.0;
 } else {
  s0+=3978.0;
  s1+=1282.0;
 }
} else {
 if(i0<15.779542922973633){
  s0+=283.0;
  s1+=117.0;
 } else {
  s0+=226.0;
  s1+=842.0;
 }
}
if(i2<3.379046678543091){
 if(i0<6.8637590408325195){
  s0+=3629.0;
  s1+=210.0;
 } else {
  s0+=1007.0;
  s1+=1932.0;
 }
} else {
 if(i3<6.276758670806885){
  s0+=109.0;
  s1+=35.0;
 } else {
  s0+=431.0;
  s1+=962.0;
 }
}
if(i2<4.758338928222656){
 if(i0<7.0543212890625){
  s0+=3662.0;
  s1+=221.0;
 } else {
  s0+=1129.0;
  s1+=2171.0;
 }
} else {
 if(i1<1.5983400344848633){
  s0+=331.0;
  s1+=742.0;
 } else {
  s0+=29.0;
  s1+=30.0;
 }
}
if(i0<6.740585803985596){
 if(i3<10.199019432067871){
  s0+=3205.0;
  s1+=83.0;
 } else {
  s0+=470.0;
  s1+=146.0;
 }
} else {
 if(i0<12.084186553955078){
  s0+=646.0;
  s1+=474.0;
 } else {
  s0+=783.0;
  s1+=2508.0;
 }
}
if(i0<7.05680513381958){
 if(i0<4.798370361328125){
  s0+=3145.0;
  s1+=60.0;
 } else {
  s0+=598.0;
  s1+=156.0;
 }
} else {
 if(i0<13.582818031311035){
  s0+=690.0;
  s1+=593.0;
 } else {
  s0+=738.0;
  s1+=2335.0;
 }
}
if(i0<10.393598556518555){
 if(i3<11.61424446105957){
  s0+=3510.0;
  s1+=178.0;
 } else {
  s0+=722.0;
  s1+=324.0;
 }
} else {
 if(i3<41.09740447998047){
  s0+=718.0;
  s1+=2451.0;
 } else {
  s0+=244.0;
  s1+=168.0;
 }
}
if(i0<7.05680513381958){
 if(i3<12.763202667236328){
  s0+=3509.0;
  s1+=115.0;
 } else {
  s0+=310.0;
  s1+=103.0;
 }
} else {
 if(i3<39.516990661621094){
  s0+=1117.0;
  s1+=2659.0;
 } else {
  s0+=259.0;
  s1+=243.0;
 }
}
if(i0<7.917067050933838){
 if(i3<6.926509857177734){
  s0+=2712.0;
  s1+=48.0;
 } else {
  s0+=1149.0;
  s1+=240.0;
 }
} else {
 if(i3<44.18832015991211){
  s0+=1107.0;
  s1+=2749.0;
 } else {
  s0+=198.0;
  s1+=112.0;
 }
}
if(i0<7.057701587677002){
 if(i0<4.525259971618652){
  s0+=3188.0;
  s1+=43.0;
 } else {
  s0+=627.0;
  s1+=164.0;
 }
} else {
 if(i0<11.958681106567383){
  s0+=549.0;
  s1+=438.0;
 } else {
  s0+=821.0;
  s1+=2485.0;
 }
}
if(i3<10.128509521484375){
 if(i3<6.35331392288208){
  s0+=2753.0;
  s1+=116.0;
 } else {
  s0+=722.0;
  s1+=238.0;
 }
} else {
 if(i0<12.001962661743164){
  s0+=949.0;
  s1+=454.0;
 } else {
  s0+=788.0;
  s1+=2295.0;
 }
}
if(i3<9.637046813964844){
 if(i2<9.151989936828613){
  s0+=3286.0;
  s1+=290.0;
 } else {
  s0+=5.0;
  s1+=35.0;
 }
} else {
 if(i0<10.148890495300293){
  s0+=943.0;
  s1+=365.0;
 } else {
  s0+=887.0;
  s1+=2504.0;
 }
}
if(i3<11.987377166748047){
 if(i0<16.855560302734375){
  s0+=3670.0;
  s1+=243.0;
 } else {
  s0+=40.0;
  s1+=288.0;
 }
} else {
 if(i3<40.301963806152344){
  s0+=1182.0;
  s1+=2405.0;
 } else {
  s0+=274.0;
  s1+=213.0;
 }
}
if(i2<4.07694673538208){
 if(i3<12.149951934814453){
  s0+=3587.0;
  s1+=380.0;
 } else {
  s0+=1200.0;
  s1+=1854.0;
 }
} else {
 if(i2<6.023122787475586){
  s0+=189.0;
  s1+=234.0;
 } else {
  s0+=248.0;
  s1+=623.0;
 }
}
if(i3<12.19261360168457){
 if(i3<6.618836402893066){
  s0+=2711.0;
  s1+=156.0;
 } else {
  s0+=992.0;
  s1+=391.0;
 }
} else {
 if(i3<39.61876678466797){
  s0+=1220.0;
  s1+=2367.0;
 } else {
  s0+=259.0;
  s1+=219.0;
 }
}
if(i3<11.257333755493164){
 if(i0<16.835834503173828){
  s0+=3509.0;
  s1+=217.0;
 } else {
  s0+=32.0;
  s1+=238.0;
 }
} else {
 if(i3<13.694995880126953){
  s0+=299.0;
  s1+=221.0;
 } else {
  s0+=1317.0;
  s1+=2482.0;
 }
}
if(i0<7.047927379608154){
 if(i3<8.890393257141113){
  s0+=2954.0;
  s1+=74.0;
 } else {
  s0+=686.0;
  s1+=168.0;
 }
} else {
 if(i3<40.16413116455078){
  s0+=1178.0;
  s1+=2789.0;
 } else {
  s0+=259.0;
  s1+=207.0;
 }
}
if(i2<5.127153396606445){
 if(i0<7.66992712020874){
  s0+=3849.0;
  s1+=290.0;
 } else {
  s0+=969.0;
  s1+=2152.0;
 }
} else {
 if(i1<0.5846142768859863){
  s0+=114.0;
  s1+=191.0;
 } else {
  s0+=187.0;
  s1+=563.0;
 }
}
if(i0<6.838334083557129){
 if(i0<4.798870086669922){
  s0+=3201.0;
  s1+=76.0;
 } else {
  s0+=549.0;
  s1+=132.0;
 }
} else {
 if(i0<11.996983528137207){
  s0+=637.0;
  s1+=473.0;
 } else {
  s0+=805.0;
  s1+=2442.0;
 }
}
if(i3<11.138038635253906){
 if(i2<4.873701095581055){
  s0+=3474.0;
  s1+=351.0;
 } else {
  s0+=92.0;
  s1+=107.0;
 }
} else {
 if(i0<12.715326309204102){
  s0+=914.0;
  s1+=574.0;
 } else {
  s0+=677.0;
  s1+=2126.0;
 }
}
if(i0<7.925726890563965){
 if(i0<4.882560729980469){
  s0+=3245.0;
  s1+=65.0;
 } else {
  s0+=650.0;
  s1+=223.0;
 }
} else {
 if(i0<13.126779556274414){
  s0+=510.0;
  s1+=477.0;
 } else {
  s0+=800.0;
  s1+=2345.0;
 }
}
if(i3<11.583112716674805){
 if(i0<12.510713577270508){
  s0+=3548.0;
  s1+=200.0;
 } else {
  s0+=50.0;
  s1+=279.0;
 }
} else {
 if(i3<40.38267517089844){
  s0+=1323.0;
  s1+=2435.0;
 } else {
  s0+=262.0;
  s1+=218.0;
 }
}
if(i2<3.6317341327667236){
 if(i0<7.046280860900879){
  s0+=3672.0;
  s1+=231.0;
 } else {
  s0+=1034.0;
  s1+=1953.0;
 }
} else {
 if(i0<14.579508781433105){
  s0+=249.0;
  s1+=96.0;
 } else {
  s0+=234.0;
  s1+=846.0;
 }
}
if(i3<9.101166725158691){
 if(i0<19.051193237304688){
  s0+=3215.0;
  s1+=169.0;
 } else {
  s0+=10.0;
  s1+=141.0;
 }
} else {
 if(i0<11.614679336547852){
  s0+=1096.0;
  s1+=451.0;
 } else {
  s0+=872.0;
  s1+=2361.0;
 }
}
if(i2<4.078885078430176){
 if(i1<-0.29747772216796875){
  s0+=729.0;
  s1+=881.0;
 } else {
  s0+=3984.0;
  s1+=1415.0;
 }
} else {
 if(i2<6.191871643066406){
  s0+=196.0;
  s1+=250.0;
 } else {
  s0+=237.0;
  s1+=623.0;
 }
}
if(i3<11.138038635253906){
 if(i0<16.880348205566406){
  s0+=3625.0;
  s1+=200.0;
 } else {
  s0+=35.0;
  s1+=266.0;
 }
} else {
 if(i3<44.78185272216797){
  s0+=1394.0;
  s1+=2482.0;
 } else {
  s0+=196.0;
  s1+=117.0;
 }
}
if(i3<12.374994277954102){
 if(i2<4.506227016448975){
  s0+=3663.0;
  s1+=429.0;
 } else {
  s0+=88.0;
  s1+=136.0;
 }
} else {
 if(i0<12.130697250366211){
  s0+=726.0;
  s1+=463.0;
 } else {
  s0+=667.0;
  s1+=2143.0;
 }
}
if(i3<10.107500076293945){
 if(i0<19.12684440612793){
  s0+=3355.0;
  s1+=174.0;
 } else {
  s0+=14.0;
  s1+=201.0;
 }
} else {
 if(i3<14.56573486328125){
  s0+=542.0;
  s1+=425.0;
 } else {
  s0+=1254.0;
  s1+=2350.0;
 }
}
if(i3<10.127470016479492){
 if(i1<0.5515608787536621){
  s0+=3445.0;
  s1+=329.0;
 } else {
  s0+=32.0;
  s1+=66.0;
 }
} else {
 if(i0<11.900396347045898){
  s0+=950.0;
  s1+=447.0;
 } else {
  s0+=796.0;
  s1+=2250.0;
 }
}
if(i0<6.872157096862793){
 if(i3<10.098575592041016){
  s0+=3291.0;
  s1+=70.0;
 } else {
  s0+=495.0;
  s1+=156.0;
 }
} else {
 if(i3<42.92589569091797){
  s0+=1216.0;
  s1+=2738.0;
 } else {
  s0+=183.0;
  s1+=166.0;
 }
}
if(i3<11.871646881103516){
 if(i3<6.013233661651611){
  s0+=2619.0;
  s1+=110.0;
 } else {
  s0+=1072.0;
  s1+=398.0;
 }
} else {
 if(i0<12.130697250366211){
  s0+=745.0;
  s1+=475.0;
 } else {
  s0+=747.0;
  s1+=2149.0;
 }
}
if(i3<11.72109603881836){
 if(i0<17.167194366455078){
  s0+=3616.0;
  s1+=220.0;
 } else {
  s0+=36.0;
  s1+=253.0;
 }
} else {
 if(i0<11.989629745483398){
  s0+=793.0;
  s1+=458.0;
 } else {
  s0+=769.0;
  s1+=2170.0;
 }
}
if(i3<11.139320373535156){
 if(i1<0.3515009880065918){
  s0+=3448.0;
  s1+=297.0;
 } else {
  s0+=120.0;
  s1+=116.0;
 }
} else {
 if(i0<10.130285263061523){
  s0+=743.0;
  s1+=345.0;
 } else {
  s0+=879.0;
  s1+=2367.0;
 }
}
if(i2<3.383451223373413){
 if(i0<7.801661491394043){
  s0+=3738.0;
  s1+=271.0;
 } else {
  s0+=918.0;
  s1+=1917.0;
 }
} else {
 if(i3<7.138869285583496){
  s0+=122.0;
  s1+=42.0;
 } else {
  s0+=421.0;
  s1+=886.0;
 }
}
if(i0<6.783822059631348){
 if(i3<8.115381240844727){
  s0+=3001.0;
  s1+=41.0;
 } else {
  s0+=784.0;
  s1+=152.0;
 }
} else {
 if(i2<-6.113883018493652){
  s0+=272.0;
  s1+=348.0;
 } else {
  s0+=1215.0;
  s1+=2502.0;
 }
}
if(i0<10.393598556518555){
 if(i3<11.501932144165039){
  s0+=3578.0;
  s1+=170.0;
 } else {
  s0+=661.0;
  s1+=330.0;
 }
} else {
 if(i3<40.16413116455078){
  s0+=745.0;
  s1+=2384.0;
 } else {
  s0+=247.0;
  s1+=200.0;
 }
}
if(i0<9.681848526000977){
 if(i0<5.819924354553223){
  s0+=3513.0;
  s1+=130.0;
 } else {
  s0+=573.0;
  s1+=319.0;
 }
} else {
 if(i3<39.95225524902344){
  s0+=810.0;
  s1+=2483.0;
 } else {
  s0+=258.0;
  s1+=229.0;
 }
}
if(i0<6.489502906799316){
 if(i3<10.820844650268555){
  s0+=3286.0;
  s1+=68.0;
 } else {
  s0+=403.0;
  s1+=114.0;
 }
} else {
 if(i0<12.080211639404297){
  s0+=648.0;
  s1+=521.0;
 } else {
  s0+=868.0;
  s1+=2407.0;
 }
}
if(i0<7.057701587677002){
 if(i1<-0.5531792640686035){
  s0+=41.0;
  s1+=28.0;
 } else {
  s0+=3780.0;
  s1+=179.0;
 }
} else {
 if(i2<9.152909278869629){
  s0+=1297.0;
  s1+=2543.0;
 } else {
  s0+=110.0;
  s1+=337.0;
 }
}
if(i0<7.928766250610352){
 if(i2<-7.1626057624816895){
  s0+=33.0;
  s1+=34.0;
 } else {
  s0+=3821.0;
  s1+=273.0;
 }
} else {
 if(i0<13.06192398071289){
  s0+=531.0;
  s1+=480.0;
 } else {
  s0+=744.0;
  s1+=2399.0;
 }
}
if(i0<10.393598556518555){
 if(i0<5.525189399719238){
  s0+=3499.0;
  s1+=89.0;
 } else {
  s0+=801.0;
  s1+=352.0;
 }
} else {
 if(i3<39.927345275878906){
  s0+=715.0;
  s1+=2400.0;
 } else {
  s0+=260.0;
  s1+=199.0;
 }
}
if(i0<6.206840991973877){
 if(i3<9.923158645629883){
  s0+=3144.0;
  s1+=56.0;
 } else {
  s0+=439.0;
  s1+=89.0;
 }
} else {
 if(i3<41.120361328125){
  s0+=1389.0;
  s1+=2773.0;
 } else {
  s0+=236.0;
  s1+=189.0;
 }
}
if(i0<7.945042610168457){
 if(i0<5.908147811889648){
  s0+=3593.0;
  s1+=131.0;
 } else {
  s0+=343.0;
  s1+=161.0;
 }
} else {
 if(i0<12.779060363769531){
  s0+=495.0;
  s1+=437.0;
 } else {
  s0+=740.0;
  s1+=2415.0;
 }
}
if(i3<11.721437454223633){
 if(i0<17.413898468017578){
  s0+=3610.0;
  s1+=225.0;
 } else {
  s0+=30.0;
  s1+=254.0;
 }
} else {
 if(i0<11.6981201171875){
  s0+=797.0;
  s1+=431.0;
 } else {
  s0+=729.0;
  s1+=2239.0;
 }
}
if(i2<4.496748447418213){
 if(i0<7.05680513381958){
  s0+=3731.0;
  s1+=216.0;
 } else {
  s0+=1089.0;
  s1+=2097.0;
 }
} else {
 if(i3<6.001718521118164){
  s0+=52.0;
  s1+=19.0;
 } else {
  s0+=327.0;
  s1+=784.0;
 }
}
if(i3<10.15129280090332){
 if(i2<4.873701095581055){
  s0+=3316.0;
  s1+=316.0;
 } else {
  s0+=63.0;
  s1+=84.0;
 }
} else {
 if(i3<15.26768970489502){
  s0+=574.0;
  s1+=510.0;
 } else {
  s0+=1227.0;
  s1+=2225.0;
 }
}
if(i3<10.127470016479492){
 if(i2<5.958948135375977){
  s0+=3323.0;
  s1+=332.0;
 } else {
  s0+=33.0;
  s1+=73.0;
 }
} else {
 if(i0<11.406106948852539){
  s0+=953.0;
  s1+=453.0;
 } else {
  s0+=820.0;
  s1+=2328.0;
 }
}
if(i0<6.711368560791016){
 if(i0<5.525619029998779){
  s0+=3485.0;
  s1+=105.0;
 } else {
  s0+=248.0;
  s1+=89.0;
 }
} else {
 if(i0<13.044380187988281){
  s0+=698.0;
  s1+=621.0;
 } else {
  s0+=757.0;
  s1+=2312.0;
 }
}
if(i3<11.72380256652832){
 if(i0<18.320770263671875){
  s0+=3633.0;
  s1+=209.0;
 } else {
  s0+=39.0;
  s1+=247.0;
 }
} else {
 if(i0<11.989629745483398){
  s0+=745.0;
  s1+=470.0;
 } else {
  s0+=740.0;
  s1+=2232.0;
 }
}
if(i0<7.259934902191162){
 if(i0<5.5486602783203125){
  s0+=3409.0;
  s1+=103.0;
 } else {
  s0+=334.0;
  s1+=138.0;
 }
} else {
 if(i0<12.945279121398926){
  s0+=639.0;
  s1+=497.0;
 } else {
  s0+=753.0;
  s1+=2442.0;
 }
}
if(i0<7.06691312789917){
 if(i3<14.045440673828125){
  s0+=3586.0;
  s1+=134.0;
 } else {
  s0+=224.0;
  s1+=91.0;
 }
} else {
 if(i3<46.44805145263672){
  s0+=1220.0;
  s1+=2793.0;
 } else {
  s0+=167.0;
  s1+=100.0;
 }
}
if(i0<7.679771423339844){
 if(i0<4.814064979553223){
  s0+=3268.0;
  s1+=77.0;
 } else {
  s0+=613.0;
  s1+=204.0;
 }
} else {
 if(i3<39.77789306640625){
  s0+=1037.0;
  s1+=2668.0;
 } else {
  s0+=245.0;
  s1+=203.0;
 }
}
if(i0<6.809746742248535){
 if(i3<8.115381240844727){
  s0+=2901.0;
  s1+=45.0;
 } else {
  s0+=816.0;
  s1+=157.0;
 }
} else {
 if(i0<12.619003295898438){
  s0+=695.0;
  s1+=558.0;
 } else {
  s0+=769.0;
  s1+=2374.0;
 }
}
if(i3<12.37750244140625){
 if(i0<17.579143524169922){
  s0+=3651.0;
  s1+=231.0;
 } else {
  s0+=41.0;
  s1+=286.0;
 }
} else {
 if(i2<5.119586944580078){
  s0+=1256.0;
  s1+=1967.0;
 } else {
  s0+=226.0;
  s1+=657.0;
 }
}
if(i2<4.492998123168945){
 if(i0<6.711368560791016){
  s0+=3645.0;
  s1+=186.0;
 } else {
  s0+=1216.0;
  s1+=2087.0;
 }
} else {
 if(i3<6.2505340576171875){
  s0+=54.0;
  s1+=21.0;
 } else {
  s0+=298.0;
  s1+=808.0;
 }
}
if(i3<11.867790222167969){
 if(i0<18.356292724609375){
  s0+=3630.0;
  s1+=227.0;
 } else {
  s0+=35.0;
  s1+=250.0;
 }
} else {
 if(i2<4.541127681732178){
  s0+=1255.0;
  s1+=1902.0;
 } else {
  s0+=259.0;
  s1+=757.0;
 }
}
if(i3<12.175309181213379){
 if(i0<18.854869842529297){
  s0+=3688.0;
  s1+=236.0;
 } else {
  s0+=28.0;
  s1+=284.0;
 }
} else {
 if(i0<11.408758163452148){
  s0+=717.0;
  s1+=398.0;
 } else {
  s0+=727.0;
  s1+=2237.0;
 }
}
if(i0<7.05680513381958){
 if(i0<4.798870086669922){
  s0+=3227.0;
  s1+=64.0;
 } else {
  s0+=570.0;
  s1+=135.0;
 }
} else {
 if(i0<13.59154987335205){
  s0+=698.0;
  s1+=579.0;
 } else {
  s0+=761.0;
  s1+=2281.0;
 }
}
if(i3<11.054208755493164){
 if(i0<18.041545867919922){
  s0+=3468.0;
  s1+=205.0;
 } else {
  s0+=30.0;
  s1+=226.0;
 }
} else {
 if(i0<10.393598556518555){
  s0+=753.0;
  s1+=331.0;
 } else {
  s0+=865.0;
  s1+=2437.0;
 }
}
if(i2<3.633369207382202){
 if(i1<-0.29811298847198486){
  s0+=682.0;
  s1+=866.0;
 } else {
  s0+=4017.0;
  s1+=1345.0;
 }
} else {
 if(i0<12.00256061553955){
  s0+=189.0;
  s1+=71.0;
 } else {
  s0+=299.0;
  s1+=846.0;
 }
}
if(i0<7.73319149017334){
 if(i0<5.258100509643555){
  s0+=3391.0;
  s1+=102.0;
 } else {
  s0+=509.0;
  s1+=202.0;
 }
} else {
 if(i0<13.064042091369629){
  s0+=535.0;
  s1+=464.0;
 } else {
  s0+=742.0;
  s1+=2370.0;
 }
}
if(i2<4.520400524139404){
 if(i3<12.445243835449219){
  s0+=3627.0;
  s1+=456.0;
 } else {
  s0+=1159.0;
  s1+=1898.0;
 }
} else {
 if(i0<13.113687515258789){
  s0+=133.0;
  s1+=60.0;
 } else {
  s0+=205.0;
  s1+=777.0;
 }
}
if(i0<7.05680513381958){
 if(i3<12.869680404663086){
  s0+=3570.0;
  s1+=111.0;
 } else {
  s0+=261.0;
  s1+=107.0;
 }
} else {
 if(i0<13.723825454711914){
  s0+=692.0;
  s1+=553.0;
 } else {
  s0+=736.0;
  s1+=2285.0;
 }
}
if(i3<10.813776016235352){
 if(i2<4.873701095581055){
  s0+=3462.0;
  s1+=311.0;
 } else {
  s0+=76.0;
  s1+=92.0;
 }
} else {
 if(i0<10.357937812805176){
  s0+=808.0;
  s1+=386.0;
 } else {
  s0+=865.0;
  s1+=2315.0;
 }
}
if(i3<11.305561065673828){
 if(i0<18.041545867919922){
  s0+=3678.0;
  s1+=216.0;
 } else {
  s0+=29.0;
  s1+=256.0;
 }
} else {
 if(i1<-0.005458831787109375){
  s0+=925.0;
  s1+=1240.0;
 } else {
  s0+=586.0;
  s1+=1385.0;
 }
}
if(i3<10.872323989868164){
 if(i0<19.051193237304688){
  s0+=3495.0;
  s1+=202.0;
 } else {
  s0+=25.0;
  s1+=218.0;
 }
} else {
 if(i0<11.296961784362793){
  s0+=838.0;
  s1+=450.0;
 } else {
  s0+=804.0;
  s1+=2283.0;
 }
}
if(i2<4.435152053833008){
 if(i0<7.05680513381958){
  s0+=3711.0;
  s1+=206.0;
 } else {
  s0+=1035.0;
  s1+=2154.0;
 }
} else {
 if(i3<6.290386199951172){
  s0+=65.0;
  s1+=30.0;
 } else {
  s0+=326.0;
  s1+=788.0;
 }
}
if(i3<11.583112716674805){
 if(i0<17.167194366455078){
  s0+=3566.0;
  s1+=223.0;
 } else {
  s0+=35.0;
  s1+=274.0;
 }
} else {
 if(i0<11.273162841796875){
  s0+=778.0;
  s1+=433.0;
 } else {
  s0+=818.0;
  s1+=2188.0;
 }
}
if(i0<7.05680513381958){
 if(i3<11.491870880126953){
  s0+=3413.0;
  s1+=100.0;
 } else {
  s0+=406.0;
  s1+=117.0;
 }
} else {
 if(i0<13.096672058105469){
  s0+=656.0;
  s1+=559.0;
 } else {
  s0+=722.0;
  s1+=2342.0;
 }
}
if(i0<7.270949363708496){
 if(i0<4.538682460784912){
  s0+=3166.0;
  s1+=53.0;
 } else {
  s0+=646.0;
  s1+=202.0;
 }
} else {
 if(i2<6.467179298400879){
  s0+=1116.0;
  s1+=2299.0;
 } else {
  s0+=216.0;
  s1+=617.0;
 }
}
if(i0<7.047927379608154){
 if(i0<4.80146598815918){
  s0+=3208.0;
  s1+=78.0;
 } else {
  s0+=535.0;
  s1+=159.0;
 }
} else {
 if(i0<12.58973503112793){
  s0+=625.0;
  s1+=505.0;
 } else {
  s0+=788.0;
  s1+=2417.0;
 }
}
if(i2<4.491660118103027){
 if(i0<6.159864902496338){
  s0+=3458.0;
  s1+=157.0;
 } else {
  s0+=1289.0;
  s1+=2206.0;
 }
} else {
 if(i3<41.33763122558594){
  s0+=302.0;
  s1+=783.0;
 } else {
  s0+=72.0;
  s1+=48.0;
 }
}
 float max_s=s0;
 int cls=1;
 if (max_s < s1) {
  max_s = s1;
  cls=2;
 }
 WRITE_IMAGE (out, POS_out_INSTANCE(x,y,z,0), cls);
}
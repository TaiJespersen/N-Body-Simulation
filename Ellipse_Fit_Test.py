#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:06:22 2024

@author: tai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

orbit = [(216.51614737053154, -2.1466043440302274), (211.9582479192419, 4.066637732081392), (207.2815747516926, 10.394853493245446), (202.46480626346005, 16.82515763969438), (197.47721544279545, 23.349704979145933), (192.32111227490384, 29.940666464643645), (187.02416619999858, 36.51107658130645), (181.55447165764943, 43.016195682450665), (175.87093449996556, 49.41478320980131), (169.95092655353977, 55.64841283072503), (163.8076220797347, 61.65358346924576), (157.48006104482684, 67.38388747481167), (151.02103738670297, 72.81500876864753), (144.48850045796272, 77.94867305580662), (137.93280111943022, 82.81162740281906), (131.44071217817446, 87.46129760445436), (125.02468032112894, 91.94670773291585), (118.68494211846334, 96.31079854162907), (112.42361786325557, 100.58857658880483), (106.28945636776314, 104.70327562559307), (100.22763632190755, 108.63684258736934), (94.20634928469377, 112.40118940970295), (88.19643471652225, 116.00409293110323), (82.17190601881018, 119.44162752206537), (76.12662705753036, 122.68887531343003), (70.09023411203766, 125.72739758512891), (64.09446201108727, 128.5708817959606), (58.140542142617754, 131.24201550564618), (52.216305610005314, 133.75514052652247), (46.30717505746748, 136.120565583175), (40.3957324475808, 138.3452178479667), (34.465361443864936, 140.4336370532729), (28.50068051907495, 142.38772061320728), (22.4898988824152, 144.20932765273068), (16.426125604937766, 145.89961642511946), (10.310953321939875, 147.46077839029897), (4.157150720839419, 148.9013493282524), (-2.0176995321683178, 150.24205749707434), (-8.205149001865351, 151.504077054202), (-14.281024239075819, 152.65811956554506), (-20.240232120619986, 153.74417304888124), (-26.090200214329695, 154.78437200345883), (-31.841179643333785, 155.79370242991982), (-37.50434587517668, 156.78285597623892), (-43.092011106244016, 157.75880923234678), (-48.61811314852233, 158.72455157789165), (-54.09636644813374, 159.68084069341677), (-59.540132287696544, 160.624851476576), (-64.96087290255362, 161.5491851449361), (-70.36527357171363, 162.4407057298384), (-75.75020932312202, 163.28047729048487), (-81.09689475920017, 164.0488159157805), (-86.37652075165693, 164.7379230050936), (-91.57734052943373, 165.34906535690425), (-96.70518802964081, 165.86197250432832), (-101.74950356567466, 166.25327448768346), (-106.69678385670463, 166.51971436290665), (-111.53918529308724, 166.66800007518148), (-116.27268487402263, 166.70848653987676), (-120.89476599680093, 166.65312235157407), (-125.40239381517216, 166.5147149000002), (-129.79187319450227, 166.30603918456913), (-134.0560033789668, 166.03919757763384), (-138.1837426798906, 165.72269681174032), (-142.16163802425172, 165.3577109697956), (-145.97906369878024, 164.9381939285036), (-149.63054056728055, 164.45638821714795), (-153.11527907054085, 163.90555340221786), (-156.43659065635777, 163.28121693606522), (-159.6004480643783, 162.58165065563944), (-162.6137514670948, 161.8073585535066), (-165.48300968333703, 160.96011864995995), (-168.2133059497944, 160.0421494536134), (-170.8076097825513, 159.05568537191763), (-173.26630993816752, 158.0033084854427), (-175.587758851264, 156.8886277880675), (-177.76955092846305, 155.71665448517606), (-179.8090276423352, 154.4940435908535), (-181.70513194296953, 153.2270561226499), (-183.45764922790082, 151.92168722282526), (-185.06631909959773, 150.58481210465106), (-186.5317796098174, 149.22145052733472), (-187.85465512971535, 147.8361495824281), (-189.0352987314292, 146.43302466565325), (-190.07359915141745, 145.0157823208888), (-190.96883547925785, 143.58772856078468), (-191.71954690218428, 142.15176787495915), (-192.32359012391163, 140.71038635260484), (-192.77772120394945, 139.26561944763867), (-193.0783709626319, 137.81844366630034), (-193.2202424991981, 136.3697729090742), (-193.19687276099424, 134.91980478887373), (-193.00083902791073, 133.4678357702088), (-192.62470558399036, 132.01113928056895), (-192.05957649401668, 130.54645615051004), (-191.29660101646505, 129.06874002893042), (-190.32799439990458, 127.57078556343447), (-189.1484439865395, 126.043203053167), (-187.75676199759576, 124.47552706697981), (-186.15712600534448, 122.85882349388585), (-184.35844478970142, 121.18928181435598), (-182.37123304462153, 119.47127413293062), (-180.20367097569414, 117.71843755765494), (-177.85899437343193, 115.95352986641643), (-175.33524433919484, 114.20938204145546), (-172.62911498499878, 112.5330429526483), (-169.74928428912742, 110.9868234977398), (-166.7223881714744, 109.62476454102043), (-163.57051551405283, 108.48011826430317), (-160.3044554133274, 107.57773599358482), (-156.92828304757828, 106.94152149795046), (-153.44329019625857, 106.59727875622664), (-149.85491471095602, 106.5723557397294), (-146.18948719664607, 106.89353426869526), (-142.55498978243963, 107.47958410384365), (-138.97161971620423, 108.26993285652044), (-135.46356543975014, 109.19245438987977), (-132.011034934658, 110.17445109834901), (-128.57255543496834, 111.11826710092103), (-125.27169377285739, 111.91553054946657), (-122.13286585174109, 112.58636131143648), (-119.14987561131386, 113.14650203850168), (-116.31094395155489, 113.60911706193352), (-113.60336395612862, 113.98566192799716), (-111.01519977277226, 114.2858502906442), (-108.53535208368828, 114.51831072107431), (-106.1538846931681, 114.69049544668805), (-103.86195073820988, 114.8088377048721), (-101.65165032415844, 114.87888434141816), (-99.51620185123747, 114.90432681281388), (-97.44899928236504, 114.88901836028023), (-95.44449108015968, 114.83593937835431), (-93.49715159639258, 114.74853136956946), (-91.60195743339158, 114.62913259653445), (-89.75477763353743, 114.47897863992328), (-87.95112025069389, 114.29935373855987), (-86.18715340282489, 114.08967944988494), (-84.45848198749573, 113.85015020868423), (-82.76057223616291, 113.58040102749226), (-81.08857358547932, 113.27952026336831), (-79.43669218563227, 112.94794712514098), (-77.79907996584657, 112.58359590937113), (-76.16841845863003, 112.18514033245548), (-74.53601163634391, 111.7523507670517), (-72.89165214138518, 111.28790147150379), (-71.22474982432374, 110.79920369613524), (-69.52695588418908, 110.30050865096277), (-67.79770143051385, 109.81422868610767), (-66.04841871674826, 109.36291512089656), (-64.29861185983907, 108.95932827109343), (-62.567479898505, 108.60406079263451), (-60.868817598729414, 108.28973851919991), (-59.205663420068504, 108.01031878260687), (-57.612205547163896, 107.8049489187839), (-56.0868073249761, 107.65265903279142), (-54.604626395086385, 107.53622770390399), (-53.127804797388045, 107.45898847687685), (-51.63349089884683, 107.44072243686044), (-50.11612934386622, 107.49883499030565), (-48.579946041231075, 107.64533116888148), (-47.03286541995639, 107.88906650066819), (-45.48454388617041, 108.23608236827926), (-43.94586281995548, 108.69109321061384), (-42.42789786014425, 109.25455507279321), (-40.940957999721796, 109.92097625575137), (-39.49290028234047, 110.67909758866422), (-38.08697747523971, 111.51394801570294), (-36.72125459607799, 112.410015593614), (-35.39009722969287, 113.35351081536342), (-34.08374620840046, 114.33514728761342), (-32.79059123376589, 115.35078005318387), (-31.498538426806764, 116.40198901195517), (-30.197623420432436, 117.49645619396937), (-28.879179072639552, 118.64750498051302), (-27.538681979475076, 119.87375432143367), (-26.177493347910964, 121.19674141324259), (-24.80318643894241, 122.63766418297175), (-23.428171086126802, 124.21449818749063), (-22.06792344432506, 125.9416256433653), (-20.738665574419333, 127.82811193588317), (-19.457348142810936, 129.87981017533107), (-18.240940678250553, 132.1030677554706), (-17.10198502025135, 134.50137658855553), (-16.05189391944844, 137.0750427684996), (-15.101293990558823, 139.82309106705307), (-14.26026118125819, 142.74403505585627), (-13.538945454449454, 145.83689238649467), (-12.947315060991484, 149.09981642963487), (-12.495783840355363, 152.53057515536614), (-12.195420866441454, 156.1260542996264), (-12.05783118999622, 159.8815029224841), (-12.094442871872582, 163.78978945808313), (-12.315131465298446, 167.84123954762856), (-12.72668000529166, 172.02460142765798), (-13.332031900722965, 176.32901357462276), (-14.130674281192174, 180.74599288467311), (-15.119885415898567, 185.27033028052423), (-16.29581668656252, 189.89985258368222), (-17.654155185839837, 194.63451446589315), (-19.190458830690854, 199.47696281028465), (-20.899464141845833, 204.43010525519145), (-22.774671446617564, 209.4954017464744), (-24.808068223641936, 214.67002821891953), (-26.992427483893156, 219.94507267875923), (-29.32515708951879, 225.3063946062018), (-31.811743617655768, 230.7400169125197), (-34.4621674088145, 236.23328608264814), (-37.28927045812658, 241.77682599607883), (-40.3066801274606, 247.36328269604977), (-43.52833790073141, 252.98482451319137), (-46.965979450681466, 258.6330547853341), (-50.62898583113733, 264.29704707998866), (-54.52340286647379, 269.96264502026486), (-58.65111170588464, 275.61232210034814), (-63.00939314689049, 281.22560047817745), (-67.59098608677002, 286.7799770170081), (-72.38465759028736, 292.25216373246485), (-77.37630090036333, 297.6192953891999), (-82.55069566995108, 302.8598104662972), (-87.89201486816776, 307.95506716091404), (-93.38465734430851, 312.8900421406946), (-99.01362196381574, 317.6542524135391), (-104.76462406061763, 322.2413329678284), (-110.62517834412338, 326.64748780363806), (-116.58299565718934, 330.8722080272443), (-122.62678901501471, 334.9164807411461), (-128.74594035711522, 338.7820824034869), (-134.9300360155897, 342.47131525610655), (-141.16854571301027, 345.9872343394391), (-147.45089262297873, 349.3341454255364), (-153.76684707577724, 352.51801297432235), (-160.10804417206447, 355.5475714857028), (-166.46916018390726, 358.43315857362194), (-172.84457384242936, 361.18486263491144), (-179.23044118075507, 363.8137708435027), (-185.62462360441393, 366.33163616883326), (-192.0266231542115, 368.75084752052857), (-198.4378327019258, 371.0843201976547), (-204.86113457070246, 373.34659063696654), (-211.30111111998525, 375.55402296627983), (-217.7632864044074, 377.7265712178627), (-224.24411300281324, 379.89395466375237), (-230.71222336448963, 382.063383062169), (-237.176665809184, 384.2347687052874), (-243.6557471972404, 386.4068142268437), (-250.16427935452433, 388.57624053247184), (-256.7038660515571, 390.7401155105987), (-263.2713328348292, 392.8964708202907), (-269.84885082035333, 395.0393110029321), (-276.4026803453036, 397.14111365290034), (-282.9112541381701, 399.13917959509376), (-289.387946402934, 400.9809874752887), (-295.84983084713497, 402.64015631015366), (-302.30630078382717, 404.1050982856085), (-308.7622595904282, 405.3722631836551), (-315.2205486481474, 406.44048195968657), (-321.68254556927855, 407.3061297551094), (-328.145902557356, 407.96482062268456), (-334.60350096039014, 408.4104114729738), (-341.0445050038442, 408.6359374114552), (-347.4542585775947, 408.63589941628067), (-353.81599753635663, 408.407750279485), (-360.1125895110933, 407.9520793360061), (-366.3273743675249, 407.271896890618), (-372.44405039092305, 406.37183927349616), (-378.44610471935533, 405.2577847764822), (-384.31633882232444, 403.93694998044566), (-390.0368926271094, 402.41823719582715), (-395.58994774360025, 400.7124682162038), (-400.9585921517098, 398.8326423459902), (-406.12775542563224, 396.7927483515484), (-411.08437301324864, 394.607673981051), (-415.8172541956514, 392.29298766372784), (-420.3169325762021, 389.86502338591674), (-424.5758519465856, 387.3409194378239), (-428.588787440527, 384.73809304696033), (-432.3584636668545, 382.0855311816669), (-435.8841350246721, 379.3979741710002), (-439.1655292607045, 376.68730359903253), (-442.20159807944526, 373.96238738610737), (-444.9888030711711, 371.24146157340584), (-447.4899197767822, 368.52853264120694), (-449.69619743904167, 365.82436685990314), (-451.5964912702709, 363.12623530757065), (-453.1782058542471, 360.4264230642755), (-454.4299026303274, 357.7096682712975), (-455.3478870850354, 354.9493644505037), (-455.9520289967516, 352.10482025570957), (-456.3114819780998, 349.13715985252134), (-456.52119927512274, 346.0533162558819), (-456.64298950557384, 342.8883250993722), (-456.70883485967784, 339.6782161174575), (-456.7323755676894, 336.4556998343779), (-456.6480819301423, 333.2592493012701), (-456.46293270105775, 330.10337858619255), (-456.17914562738514, 326.99691510722386), (-455.79613284873415, 323.9414115133632), (-455.31559169186625, 320.9280576700191), (-454.7392902475224, 317.94435722285334), (-454.07783343582224, 314.9724540929643), (-453.28426261294544, 312.0060764899343), (-452.33676024653107, 309.003100030125), (-451.24744164209085, 305.9635334584541), (-450.02965199246694, 302.8902854815418), (-448.6982009815046, 299.78591390351096), (-447.267178585211, 296.6496019993595), (-445.7466031228711, 293.4772598645133), (-444.14156424463573, 290.26366363815754), (-442.45337265766085, 287.00379656485774), (-440.6785297384076, 283.696905510722), (-438.81292650488325, 280.3379281371203), (-436.86752019318476, 276.9125449825558), (-434.82464614763643, 273.428413941858), (-432.6830571147646, 269.88811553328145), (-430.440595886878, 266.2926191126968), (-428.09436021458606, 262.6427738519744), (-425.6407566003377, 258.94000749073314), (-423.0755442791712, 255.1871339346082), (-420.4176792369824, 251.38355045835016), (-417.6615187496919, 247.53987050010082), (-414.8054035270743, 243.67990072189184), (-411.8757399887371, 239.85297744329688), (-408.94297887209336, 236.08009295261363), (-406.0412370708803, 232.3452659336204), (-403.180374242024, 228.6345136449327), (-400.35269098271954, 224.94463791646723), (-397.56673079337577, 221.2567626922205), (-394.8207112157063, 217.56212257181178), (-392.1127306246383, 213.85394449824), (-389.4400142773242, 210.12837373111518), (-386.799015617398, 206.38313768368576), (-384.1845077382457, 202.61767727914898), (-381.5892853619073, 198.83265241567773), (-379.0039434571861, 195.02955475540907), (-376.4174325242699, 191.20845317953373), (-373.81551423199244, 187.3712997583808), (-371.18182725841785, 183.51951253373795), (-368.4986672725614, 179.65543384492696), (-365.74972376056627, 175.77562360558355), (-362.94663794462843, 171.8408041377626), (-360.22896352658717, 167.9337193318891), (-357.5772185769197, 164.05552412946656), (-354.97107781917924, 160.1990865160288), (-352.390282438686, 156.3557621360139), (-349.8165403201881, 152.51545124909305), (-347.2341019883109, 148.66761685467677), (-344.6297866420698, 144.80196824144875), (-341.9926933995939, 140.9088667370383), (-339.30610894722474, 136.98136927148167), (-336.5627097069496, 133.0107088830939), (-333.7564714706519, 128.98949894618139), (-330.8820441349137, 124.91108618723605), (-327.9344737962728, 120.76939790055872), (-324.90897601309536, 116.55878628446573), (-321.80075383832514, 112.27387358798012), (-318.6048507283814, 107.90939618491414), (-315.3160299426621, 103.4600434300848), (-311.9286749262222, 98.92028605917577), (-308.4367074907411, 94.28418708281343), (-304.8335226356789, 89.54518666492936), (-301.1114792604222, 84.69492623344814), (-297.2627970665022, 79.724830571151), (-293.27907127793196, 74.62481295869418), (-289.15126213181264, 69.3828891857709), (-284.8697035639863, 63.984763128039475), (-280.42665780056774, 58.41110832847735), (-275.8184556898085, 52.64026489896211), (-271.056033950779, 46.643323273573806), (-266.20777510672457, 40.46380802702459), (-261.26555702382603, 34.097256308692614), (-256.2200165556284, 27.537359247577186), (-251.06024790194633, 20.78198798157181), (-245.77748031909772, 13.840718845065378), (-240.38231265415783, 6.739341194040185), (-234.9101751466445, -0.5184229510250757), (-229.36620583023446, -7.955252350191064), (-223.74011796220555, -15.593942729986551), (-218.0257415751511, -23.45673331651972), (-212.2290728169102, -31.566611928451362), (-206.36647210347223, -39.94430733839739), (-200.45265186600483, -48.60484424969299), (-194.49011587709526, -57.561570196784984), (-188.46720130225935, -66.82766209094905), (-182.3590115144462, -76.41588422312864), (-176.12883797901912, -86.32845443799243), (-169.74036656686363, -96.54051808083064), (-163.18402758804868, -106.9669439902874), (-156.55603256976048, -117.51699767482563), (-149.94764005626783, -128.1633926995204), (-143.4217480046711, -138.88035526326712), (-137.11333539284962, -149.61503664319818), (-131.02134507145104, -160.2836139960043), (-125.11897250912654, -170.86769341224814), (-119.39139288400123, -181.34310879797428), (-113.83117448306717, -191.64283281155494), (-108.37219895349455, -201.6214485003381), (-102.98628434679787, -211.20956303775952), (-97.66157112071357, -220.44169105445923), (-92.3768865886142, -229.3530883281054), (-87.11551338213363, -237.9736214811992), (-81.86675905055147, -246.31940735894702), (-76.63796150025775, -254.39000589888897), (-71.44330219490287, -262.2030354121879), (-66.2876931346918, -269.7808275886565), (-61.17315792452651, -277.14462963668296), (-56.10063608459703, -284.31419920296383), (-51.07059138283336, -291.3082307776639), (-46.08333470946853, -298.1451010575037), (-41.14091115085334, -304.84359112249655), (-36.2444945672324, -311.4240049872584), (-31.397277842104963, -317.90771573366123), (-26.609483254425225, -324.3117497786651), (-21.91177206650371, -330.60679846918555), (-17.251389255053237, -336.71299688548993), (-12.58624544317779, -342.63461898444615), (-7.882844759957999, -348.3671096505889), (-3.1176518472099044, -353.871800124686), (1.6977176296271947, -359.09434190020335), (6.536996201252789, -364.00279040616795), (11.373920306209026, -368.5743809117245), (16.17374421794781, -372.7923740025959), (20.896438689652037, -376.65515531365145), (25.50851598493294, -380.1746264055643), (29.985375050005356, -383.3681519842671), (34.31022710904282, -386.25302888748985), (38.47136980845483, -388.844176224166), (42.45986173638239, -391.15312780211127), (46.26892127294701, -393.18864511554654), (49.8930609375538, -394.9564130064131), (53.32573645476918, -396.4592619833088), (56.56155238559741, -397.69731541254214), (59.59515799778616, -398.6679699524474), (62.4214349230612, -399.36602156774245), (65.03595118024475, -399.78406749145853), (67.4361865078421, -399.91353479277245), (69.62188980900278, -399.74522753027975), (71.59879039387887, -399.26982282352014), (73.4090179189528, -398.4776094466618), (75.1026538645175, -397.43137399179375), (76.68634870801232, -396.17230922157114), (78.18062705113074, -394.7389792464758), (79.62399197230536, -393.1803991538176), (81.0534454281985, -391.49084397122664), (82.4656577889208, -389.6557832974937), (83.84484709346312, -387.6611686316804), (85.17082022493263, -385.49452593331216), (86.42258134370536, -383.14396145065757), (87.57758713540444, -380.598090349761), (88.61317064579198, -377.8450587603866), (89.50659195274656, -374.871853797537), (90.23582993279183, -371.66365473264017), (90.77851135746452, -368.20322527901135), (91.11362800207931, -364.47060792827807), (91.22393029275463, -360.4433324552573), (91.0989146835308, -356.10040536553225), (90.7381982687796, -351.43054513820465), (90.14681142164005, -346.4464615871883), (89.32163328841821, -341.19145178122267), (88.25242723071013, -335.72616946295756), (86.93342526461889, -330.1495785310658), (85.56232166244057, -324.54258927924656), (84.13334228581115, -318.88502301363593), (82.62348323707882, -313.1688367494532), (81.00682516881658, -307.3895409547591), (79.25256668546227, -301.54078081176647), (77.32910706659058, -295.62443847082), (75.22005369623858, -289.65315846499396), (72.926815740468, -283.62598581754776), (70.4374962263273, -277.5217744577928), (67.7253576942576, -271.3212578861746), (64.7717539930452, -265.01643685951797), (61.58888089778571, -258.5997043315992), (58.20190728983087, -252.05111219720405), (54.62059586962865, -245.36788072804612), (50.83535275142225, -238.57169327606954), (46.82051398846069, -231.69996905105063), (42.58416737161083, -224.83514897148657), (38.1372613868566, -218.06653526684408), (33.48028642108379, -211.47889071746278), (28.608595762525173, -205.1275077401603), (23.540120171294195, -199.0520719400107), (18.406727750013275, -193.24569162123476), (13.283651216977665, -187.73758770639645), (8.213165474414224, -182.5231030986805), (3.225953339316618, -177.58279999735203), (-1.655747532691346, -172.89455270213762), (-6.414041362962386, -168.435016447852), (-11.034976899531227, -164.18459746070968), (-15.509057626850007, -160.12883823177327), (-19.83011650938454, -156.25835905231), (-23.996132118691015, -152.56822528880033), (-28.00833819070437, -149.06007951579946), (-31.86734968727778, -145.7399415145263), (-35.56615260231182, -142.61365825230862), (-39.091542598762935, -139.67950490447726), (-42.43276493147618, -136.9244949407147), (-45.584356336172746, -134.3311661102048), (-48.543577902187145, -131.8817943469929), (-51.308948915659236, -129.56123120709708), (-53.87791634057449, -127.35716693791159), (-56.247451961927986, -125.26101678807025), (-58.41443270062622, -123.26532689495806), (-60.37617958002131, -121.36715468161971), (-62.13106102724312, -119.5655988113327), (-63.678755676156364, -117.86088386400746), (-65.0204260932325, -116.25349414301628), (-66.15711771074615, -114.74298824093444), (-67.08953436105057, -113.33040160310892), (-67.81706650214213, -112.01374808112132), (-68.33843399893767, -110.7897547627815), (-68.65117409717266, -109.65789456032778), (-68.7519187775751, -108.61877261499971), (-68.63596026549926, -107.67443840721663), (-68.2989427024262, -106.82837222403367), (-67.73821427504937, -106.08533698856797), (-66.95243709700874, -105.45065284702756), (-65.94160248518898, -104.92941245480627), (-64.70781488042995, -104.52461695016606), (-63.25232850036372, -104.24128199719604), (-61.576843330564664, -104.08115702307805), (-59.68305734882062, -104.04454651939007), (-57.57290027468854, -104.12893164967433), (-55.24776518433771, -104.33102988972479), (-52.709063641527955, -104.6437595845153), (-49.95810753001154, -105.05910183810091), (-46.99455947108634, -105.56511317209299), (-43.81946692265845, -106.14674006964562), (-40.4354536220711, -106.78538413311718), (-36.84840267645253, -107.45822588496856), (-33.0719513322271, -108.13995925839718), (-29.13304736563185, -108.80565053098914), (-25.073440737221095, -109.43957749921786), (-20.937325908271, -110.04315857648439), (-16.75384724232994, -110.62642082373905), (-12.53490016316923, -111.19814586360404), (-8.28217977257422, -111.76165979762375), (-3.995582332217282, -112.31628287618126), (0.31796303744713494, -112.85978247830387), (4.636929581814979, -113.40228105575727), (8.952829404710052, -113.97193427045826), (13.276377277075904, -114.58410263291644), (17.622045396715013, -115.24246325889422), (22.00227459445643, -115.94518651518352), (26.426703187189737, -116.68832408530473), (30.90237091070403, -117.46724925483942), (35.43410225246562, -118.27730560390636), (40.024899544312326, -119.11416310078384), (44.676550697243435, -119.97464870421061), (49.38959345444687, -120.85631811914728), (54.164081996970054, -121.75701842992149), (59.00075364653565, -122.67425544837035), (63.899774056901535, -123.60757322695352), (68.8617925676106, -124.55737984626384), (73.88811649184497, -125.52513874038497), (78.98111920855095, -126.51358137157392), (84.1440692144067, -127.52815541388357), (89.38183797450083, -128.57872008988616), (94.70223399527818, -129.68111147396573), (100.11766608156417, -130.8621603300624), (105.64309953189492, -132.16109206348287), (111.28638513480115, -133.633742143866), (117.02427194219553, -135.33648100509288), (122.82291171474547, -137.28224261088954), (128.68641757499094, -139.45740567714975), (134.6437514204373, -141.84824753092423), (140.73098919400616, -144.44756808111813), (146.98444525516106, -147.2513654951223), (153.4378028914832, -150.25608422047583), (160.12277081520233, -153.4569308834266), (167.06971207805293, -156.84697717490855), (174.30435650790116, -160.41309258659746), (181.84959297001265, -164.12326487178373), (189.70055111615127, -167.9225074684917), (197.8162119017783, -171.79654588815563), (206.21050508107737, -175.75996093885652), (214.91936246884597, -179.81976603372803), (223.9624644055467, -183.99368514659972), (233.01900706027783, -188.21478941185362), (242.13048582728806, -192.46887796465018), (251.37134408947279, -196.78433437866158), (260.82167398444625, -201.20104249756852), (270.4931066804055, -205.76100673736235), (280.21582229774367, -210.3126233611953), (289.90616044188744, -214.65174553448696), (299.5065419131852, -218.75370581694398), (309.03121792356905, -222.6308403032903), (318.49603453214377, -226.35903252277683), (327.7937181042295, -229.9876026104081), (336.8472723797159, -233.52479561096948), (345.64029681327645, -236.93080375933235), (354.2083581701669, -240.20925315660924), (362.6136609517128, -243.3646598481966), (370.9148968937831, -246.36094861023196), (379.1058293981127, -249.13991199603115), (387.14529303461114, -251.68023192696052), (394.98097099606247, -254.00259745889255), (402.5851234889464, -256.15654828157574), (409.94967805854077, -258.1926161847577), (417.0741765006304, -260.15322490408), (423.95967392443595, -262.0707205645565), (430.60723732932183, -263.96775296174326), (437.0187427929152, -265.8589718587992), (443.19776580934723, -267.753871256025), (449.14665556661373, -269.66263063759357), (454.87076196733517, -271.5913436892842), (460.3828000598171, -273.5324076964997), (465.6815173618671, -275.4903025017232), (470.7562384199971, -277.46614323744944), (475.5808233679793, -279.438977092129), (480.1292834950344, -281.32282492111943), (484.4186489477576, -283.1002031745041), (488.46435656297473, -284.7667026670624), (492.28054766513293, -286.32660950746964), (495.8818370108698, -287.7810982767601), (499.28187393379477, -289.1327075536352), (502.4931279101998, -290.3849932147054), (505.5266931369707, -291.5382531125899), (508.39213422983465, -292.59762577195295), (511.09752389909806, -293.56893916912037), (513.6497401710673, -294.4576351030245), (516.056129090699, -295.27628448359206), (518.3204000483831, -296.0209814616826), (520.4446122055456, -296.69520443600646), (522.4298273911174, -297.3022548141777), (524.2761545185192, -297.8452082072858), (525.9827804712968, -298.3268727134982), (527.5479864877884, -298.74974122121245), (528.9665350888038, -299.12022018473465), (530.2348209664262, -299.440175131804), (531.3483780311775, -299.71091584588135), (532.3019792409326, -299.93355556594963), (533.0897486588035, -300.10787672641794), (533.7054470440517, -300.2327414689801), (534.1428291677519, -300.3061279476418), (534.3960838730941, -300.3266298386464), (534.459749901485, -300.29137545678793), (534.3508436651043, -300.2348168815747), (534.057015601161, -300.156563942731), (533.5652625317557, -300.0517019231384), (532.9280067334239, -299.91829803761624), (532.1401794854225, -299.7600241320969), (531.1963448571282, -299.5811856886105), (530.0903247572453, -299.3868288054864), (528.815426109263, -299.1829660052589), (527.3645606607546, -298.97683625880376), (525.7305213337206, -298.77723614152626), (523.9065392351843, -298.5949279787621), (521.8873351404186, -298.4431050977491), (519.7387346668796, -298.2431633718496), (517.4571829687757, -297.99502108202614), (515.038835770971, -297.6983945810535), (512.4795547344119, -297.3527457329498), (509.7747783567063, -296.9545327757821), (506.91979179256566, -296.50306850559184), (503.90964343101007, -295.99674151176185), (500.7392280534137, -295.43349950404206), (497.40334569292327, -294.8108376564065), (493.89675439146544, -294.1257890776977), (490.21419194719476, -293.3749115326902), (486.35039609949104, -292.55426118211943), (482.2994873041842, -291.6601070856894), (478.05611395202834, -290.6873858963135), (473.6147787035716, -289.63025720978357), (468.9699070457686, -288.48004270299697), (464.1153989949202, -287.22864369821315), (459.0449809154805, -285.8657535013085), (453.7534959187638, -284.3776963176782), (448.24032203323134, -282.74723674496425), (442.5133123835701, -280.95754552471476), (436.58638487768593, -278.99975717175994), (430.46871505482403, -276.8780125094156), (424.16538287337846, -274.5978386337462), (417.67790038715765, -272.1669766235567), (411.0054239248706, -269.5960162373257), (404.14542281319444, -266.9027686068009), (397.0958849137732, -264.1037229097657), (389.846510925558, -261.20395423524667), (382.39606824244765, -258.1724906095742), (374.9135518195766, -255.04012984991058), (367.4191462388685, -251.8955734963786), (360.072962588823, -248.720162932734), (352.95070430176696, -245.50648756368727), (346.0363104294852, -242.25331882114475), (339.30341856642656, -238.9555715276519), (332.72835363008596, -235.60412782961188), (326.2937816298873, -232.18381136605498), (319.98540815284815, -228.6781120848457), (313.78628047141547, -225.07340302777286), (307.6865948192718, -221.35576823598032), (301.6782403022595, -217.5134776775399), (295.7545015185705, -213.53719939884724), (289.91201956356826, -209.41673474040047), (284.1446851631698, -205.15043413029963), (278.44706778467497, -200.7448485982648), (272.82503872952776, -196.22493670498787), (267.32682741686995, -191.61447764064), (261.9797963846778, -186.90044793861983), (256.7885756584265, -182.07192829099793), (251.75852503689546, -177.1225789470225), (246.88626717031622, -172.04929718117705), (242.17309622081157, -166.8494856837471), (237.6208992771562, -161.52012629380758), (233.2307317778501, -156.05880377469813), (229.00264078905894, -150.46318156059698), (224.93504710204334, -144.73209888502498), (221.02745925349888, -138.86735425586494), (217.27672407643954, -132.87766354130827), (213.67631363775314, -126.77772432354614), (210.22307097054204, -120.58681655502068), (206.9222860832832, -114.32473754995624), (203.7905776447938, -108.01617737619215), (200.83637491437716, -101.69129756521481), (198.05841072262888, -95.36605553354161), (195.4551244635185, -89.0441861116855), (193.02668284251362, -82.72472682176831), (190.77312368161424, -76.40355678412557), (188.6983865507435, -70.07522655867851), (186.80579850157866, -63.73345995079577), (185.09866560540368, -57.370636878651084), (183.57936111848048, -50.97823307366958), (182.24774936273778, -44.548640854404866), (181.0989262005775, -38.0806674329058), (180.12234142423802, -31.590530859546423), (179.31982008378185, -25.119141570680767), (178.70981562607201, -18.70693739093662), (178.30151260558117, -12.381721372145396), (178.0884635568005, -6.159386388949302)]


x = np.array([i[0] for i in orbit])
y = np.array([i[1] for i in orbit])

A = np.stack([x**2, x * y, y**2,x , y]).T
b = np.ones_like(x)
w = np.linalg.lstsq(A, b)[0].squeeze()

A, B, C, D, E = w

# Form the matrix M
M = np.array([[A, B / 2], [B / 2, C]])

# Find the eigenvalues and eigenvectors of M to determine the ellipse axes and orientation
eigvals, eigvecs = eig(M)  # Only the upper-left 2x2 part matters for the ellipse
print(f"EIGVALS {eigvals}")
# Sort eigenvalues (larger eigenvalue is the semi-major axis)
eigvals = np.abs(eigvals)  # Eigenvalues should be positive, so take absolute values
eigvals = np.sort(eigvals)

print(f"EIGVALS2 {eigvals}")
a = np.sqrt(1 / eigvals[0])  # Semi-major axis (larger eigenvalue)
b = np.sqrt(1 / eigvals[1])  # Semi-minor axis (smaller eigenvalue)
print(f"A {A} B {B}")
# Compute the eccentricity of the ellipse
eccentricity = np.sqrt(1 - (b**2 / a**2))# Example data points (x, y)
print(f"ECCENTRICITY {eccentricity}")


xlin = np.linspace(-1000, 1000, 300)
ylin = np.linspace(-1000, 1000, 300)
X, Y = np.meshgrid(xlin, ylin)

Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 #+ w[3]*X + w[4]*Y


fig, axe = plt.subplots(figsize=(8, 6))
plt.grid(False, which='both', linestyle='--', color='gray', alpha=0.5)
axe.scatter(x, y, label='Particle Orbit', color='firebrick')
contour = axe.contour(X, Y, Z, [1])

from matplotlib.lines import Line2D

# Create custom legend entry for the contour line
ellipse_line = Line2D([0], [0], color='purple', lw=2)  # Adjust color/lw as needed
axe.legend([ellipse_line, plt.Line2D([], [], marker='o', color='firebrick', label='Particle Orbit')],
           ['Fitted Ellipse', 'Particle Orbit'], loc='upper right')
plt.show()


'''
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])

x = x*10
y=y*10

A = np.stack([x**2, x * y, y**2, x, y]).T
#b= np.zeros(len(data))
b = np.ones_like(x)
w = np.linalg.lstsq(A, b)[0].squeeze()

xlin = np.linspace(-100, 100, 300)
ylin = np.linspace(-100, 100, 300)
X, Y = np.meshgrid(xlin, ylin)

Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 #+ w[3]*X + w[4]*Y

fig, axe = plt.subplots()
axe.scatter(x, y)
axe.contour(X, Y, Z, [1])
axe.show()
'''
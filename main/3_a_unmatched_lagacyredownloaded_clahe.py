import os
import pandas as pd
import requests
from astropy.io import fits
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Parameters and paths ---
BASE_DIR = "/data/rajit"
META_CSV = os.path.join(BASE_DIR, "gz3d_metadata.csv")
OUT_FITS_DIR = os.path.join(BASE_DIR, "DATA_LEGACY_DIR")
BEST_DIR = os.path.join(BASE_DIR, "DATA_LEGACY_CONVERTED_BW", "BEST")
os.makedirs(OUT_FITS_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)

# --- Skipped log string (copy-paste from prompt) ---
SKIPPED_LOG = """
Skipped mask folders and reasons:
gz3d_1-452163_19_14732475: No matching image for MANGAID 1-452163
gz3d_1-212496_37_14720400: No matching image for MANGAID 1-212496
gz3d_1-538890_37_14738951: No matching image for MANGAID 1-538890
gz3d_1-39830_37_14711213: No matching image for MANGAID 1-39830
gz3d_1-633549_127_14744168: No matching image for MANGAID 1-633549
gz3d_1-118373_127_14716197: No matching image for MANGAID 1-118373
gz3d_1-117342_127_14716118: No matching image for MANGAID 1-117342
gz3d_1-598372_61_14742152: No matching image for MANGAID 1-598372
gz3d_1-341121_127_14726411: No matching image for MANGAID 1-341121
gz3d_1-405707_127_14729851: No matching image for MANGAID 1-405707
gz3d_1-404199_127_14729754: No matching image for MANGAID 1-404199
gz3d_1-123901_91_14716440: No matching image for MANGAID 1-123901
gz3d_1-364246_61_14727826: No matching image for MANGAID 1-364246
gz3d_1-569442_61_14740724: No matching image for MANGAID 1-569442
gz3d_1-414518_127_14730279: No matching image for MANGAID 1-414518
gz3d_1-563754_61_14740126: No matching image for MANGAID 1-563754
gz3d_1-261709_127_14722670: No matching image for MANGAID 1-261709
gz3d_1-50666_127_14711920: No matching image for MANGAID 1-50666
gz3d_1-585433_37_14741204: No matching image for MANGAID 1-585433
gz3d_1-225617_61_14720968: No matching image for MANGAID 1-225617
gz3d_1-101451_127_14715482: No matching image for MANGAID 1-101451
gz3d_1-584637_37_14741113: No matching image for MANGAID 1-584637
gz3d_1-111683_61_14715943: No matching image for MANGAID 1-111683
gz3d_1-39811_37_14711212: No matching image for MANGAID 1-39811
gz3d_1-367701_91_14728011: No matching image for MANGAID 1-367701
gz3d_1-601160_19_14742174: No matching image for MANGAID 1-601160
gz3d_1-625212_91_14743617: No matching image for MANGAID 1-625212
gz3d_1-581301_61_14741078: No matching image for MANGAID 1-581301
gz3d_1-110335_61_14715898: No matching image for MANGAID 1-110335
gz3d_1-104661_37_14715697: No matching image for MANGAID 1-104661
gz3d_1-602390_61_14742214: No matching image for MANGAID 1-602390
gz3d_1-52782_127_14712110: No matching image for MANGAID 1-52782
gz3d_1-98691_127_14715368: No matching image for MANGAID 1-98691
gz3d_1-358832_19_14727366: No matching image for MANGAID 1-358832
gz3d_1-316773_37_14725042: No matching image for MANGAID 1-316773
gz3d_1-42688_19_14711395: No matching image for MANGAID 1-42688
gz3d_1-212366_61_14720374: No matching image for MANGAID 1-212366
gz3d_1-586873_61_14741332: No matching image for MANGAID 1-586873
gz3d_1-455346_37_14732719: No matching image for MANGAID 1-455346
gz3d_1-33680_127_14710736: No matching image for MANGAID 1-33680
gz3d_1-110159_127_14715888: No matching image for MANGAID 1-110159
gz3d_1-40697_37_14711259: No matching image for MANGAID 1-40697
gz3d_1-312820_127_14724833: No matching image for MANGAID 1-312820
gz3d_1-268069_37_14722873: No matching image for MANGAID 1-268069
gz3d_1-528694_37_14738263: No matching image for MANGAID 1-528694
gz3d_1-594150_127_14742032: No matching image for MANGAID 1-594150
gz3d_1-603378_61_14742241: No matching image for MANGAID 1-603378
gz3d_1-99770_37_14715405: No matching image for MANGAID 1-99770
gz3d_1-95990_127_14715267: No matching image for MANGAID 1-95990
gz3d_1-228870_127_14721117: No matching image for MANGAID 1-228870
gz3d_1-580438_127_14741060: No matching image for MANGAID 1-580438
gz3d_1-46494_127_14711606: No matching image for MANGAID 1-46494
gz3d_1-635712_61_14744287: No matching image for MANGAID 1-635712
gz3d_1-401759_37_14729572: No matching image for MANGAID 1-401759
gz3d_1-605248_61_14742404: No matching image for MANGAID 1-605248
gz3d_1-105626_61_14715751: No matching image for MANGAID 1-105626
gz3d_1-12135_61_14709940: No matching image for MANGAID 1-12135
gz3d_1-111522_61_14715937: No matching image for MANGAID 1-111522
gz3d_1-102936_37_14715593: No matching image for MANGAID 1-102936
gz3d_1-117312_37_14716115: No matching image for MANGAID 1-117312
gz3d_1-51336_61_14711994: No matching image for MANGAID 1-51336
gz3d_1-479471_127_14734331: No matching image for MANGAID 1-479471
gz3d_1-624911_61_14743593: No matching image for MANGAID 1-624911
gz3d_1-117104_37_14716105: No matching image for MANGAID 1-117104
gz3d_1-633856_127_14744211: No matching image for MANGAID 1-633856
gz3d_1-99133_127_14715385: No matching image for MANGAID 1-99133
gz3d_1-586244_61_14741299: No matching image for MANGAID 1-586244
gz3d_1-41093_127_14711291: No matching image for MANGAID 1-41093
gz3d_1-111824_127_14715950: No matching image for MANGAID 1-111824
gz3d_1-244980_37_14721806: No matching image for MANGAID 1-244980
gz3d_1-105793_91_14715763: No matching image for MANGAID 1-105793
gz3d_1-101902_37_14715519: No matching image for MANGAID 1-101902
gz3d_1-97312_37_14715326: No matching image for MANGAID 1-97312
gz3d_1-353747_127_14727080: No matching image for MANGAID 1-353747
gz3d_1-592984_127_14741834: No matching image for MANGAID 1-592984
gz3d_1-580628_127_14741066: No matching image for MANGAID 1-580628
gz3d_1-110304_19_14715895: No matching image for MANGAID 1-110304
gz3d_1-583694_37_14741104: No matching image for MANGAID 1-583694
gz3d_1-40870_37_14711279: No matching image for MANGAID 1-40870
gz3d_1-583321_127_14741098: No matching image for MANGAID 1-583321
gz3d_1-40540_37_14711252: No matching image for MANGAID 1-40540
gz3d_1-493479_19_14735513: No matching image for MANGAID 1-493479
gz3d_1-523506_127_14737936: No matching image for MANGAID 1-523506
gz3d_1-379196_127_14728482: No matching image for MANGAID 1-379196
gz3d_1-594167_37_14742039: No matching image for MANGAID 1-594167
gz3d_1-443723_37_14731889: No matching image for MANGAID 1-443723
gz3d_1-451006_37_14732403: No matching image for MANGAID 1-451006
gz3d_1-636700_61_14744308: No matching image for MANGAID 1-636700
gz3d_1-102392_91_14715561: No matching image for MANGAID 1-102392
gz3d_1-40778_37_14711269: No matching image for MANGAID 1-40778
gz3d_1-108127_127_14715822: No matching image for MANGAID 1-108127
gz3d_1-463648_37_14733469: No matching image for MANGAID 1-463648
gz3d_1-592695_91_14741805: No matching image for MANGAID 1-592695
gz3d_1-588444_61_14741409: No matching image for MANGAID 1-588444
gz3d_1-474153_127_14734112: No matching image for MANGAID 1-474153
gz3d_1-118034_61_14716169: No matching image for MANGAID 1-118034
gz3d_1-198387_37_14719745: No matching image for MANGAID 1-198387
gz3d_1-101900_37_14715518: No matching image for MANGAID 1-101900
gz3d_1-363493_91_14727782: No matching image for MANGAID 1-363493
gz3d_1-508691_37_14736868: No matching image for MANGAID 1-508691
gz3d_1-49973_127_14711842: No matching image for MANGAID 1-49973
gz3d_1-120509_37_14716272: No matching image for MANGAID 1-120509
gz3d_1-335253_127_14726037: No matching image for MANGAID 1-335253
gz3d_1-103659_127_14715634: No matching image for MANGAID 1-103659
gz3d_1-637279_91_14744317: No matching image for MANGAID 1-637279
gz3d_1-598283_61_14742150: No matching image for MANGAID 1-598283
gz3d_1-594175_127_14742041: No matching image for MANGAID 1-594175
gz3d_1-231847_37_14721236: No matching image for MANGAID 1-231847
gz3d_1-103247_127_14715610: No matching image for MANGAID 1-103247
gz3d_1-444500_61_14731979: No matching image for MANGAID 1-444500
gz3d_1-383145_127_14728701: No matching image for MANGAID 1-383145
gz3d_1-52554_91_14712091: No matching image for MANGAID 1-52554
gz3d_1-39500_127_14711195: No matching image for MANGAID 1-39500
gz3d_1-542578_19_14739174: No matching image for MANGAID 1-542578
gz3d_1-104439_61_14715678: No matching image for MANGAID 1-104439
gz3d_1-601940_19_14742193: No matching image for MANGAID 1-601940
gz3d_1-558624_127_14739864: No matching image for MANGAID 1-558624
gz3d_1-42373_37_14711373: No matching image for MANGAID 1-42373
gz3d_1-104998_61_14715724: No matching image for MANGAID 1-104998
gz3d_1-211547_127_14720317: No matching image for MANGAID 1-211547
gz3d_1-105330_127_14715741: No matching image for MANGAID 1-105330
gz3d_1-101369_91_14715477: No matching image for MANGAID 1-101369
gz3d_1-266647_37_14722824: No matching image for MANGAID 1-266647
gz3d_1-40700_37_14711260: No matching image for MANGAID 1-40700
gz3d_1-51125_127_14711977: No matching image for MANGAID 1-51125
gz3d_1-50577_37_14711913: No matching image for MANGAID 1-50577
gz3d_1-592948_127_14741833: No matching image for MANGAID 1-592948
gz3d_1-591678_91_14741680: No matching image for MANGAID 1-591678
gz3d_1-563967_37_14740132: No matching image for MANGAID 1-563967
gz3d_1-52169_61_14712077: No matching image for MANGAID 1-52169
gz3d_1-541213_127_14739091: No matching image for MANGAID 1-541213
gz3d_1-456521_61_14732878: No matching image for MANGAID 1-456521
gz3d_1-317878_37_14725092: No matching image for MANGAID 1-317878
gz3d_1-183845_127_14719187: No matching image for MANGAID 1-183845
gz3d_1-633279_61_14744131: No matching image for MANGAID 1-633279
gz3d_1-200103_127_14719827: No matching image for MANGAID 1-200103
gz3d_1-42274_37_14711364: No matching image for MANGAID 1-42274
gz3d_1-547339_127_14739437: No matching image for MANGAID 1-547339
gz3d_1-584810_127_14741134: No matching image for MANGAID 1-584810
gz3d_1-102356_127_14715558: No matching image for MANGAID 1-102356
gz3d_1-428368_127_14731166: No matching image for MANGAID 1-428368
gz3d_1-590975_127_14741588: No matching image for MANGAID 1-590975
gz3d_1-601156_91_14742172: No matching image for MANGAID 1-601156
gz3d_1-582379_61_14741083: No matching image for MANGAID 1-582379
gz3d_1-593748_127_14741947: No matching image for MANGAID 1-593748
gz3d_1-485789_37_14734807: No matching image for MANGAID 1-485789
gz3d_1-558592_91_14739858: No matching image for MANGAID 1-558592
gz3d_1-44990_127_14711537: No matching image for MANGAID 1-44990
gz3d_1-101985_37_14715527: No matching image for MANGAID 1-101985
gz3d_1-100885_61_14715449: No matching image for MANGAID 1-100885
gz3d_1-42801_37_14711409: No matching image for MANGAID 1-42801
gz3d_1-585715_37_14741275: No matching image for MANGAID 1-585715
gz3d_1-582602_61_14741091: No matching image for MANGAID 1-582602
gz3d_1-399542_37_14729452: No matching image for MANGAID 1-399542
gz3d_1-328664_127_14725659: No matching image for MANGAID 1-328664
gz3d_1-41749_127_14711321: No matching image for MANGAID 1-41749
gz3d_1-101476_61_14715483: No matching image for MANGAID 1-101476
gz3d_1-103960_127_14715653: No matching image for MANGAID 1-103960
gz3d_1-102750_19_14715577: No matching image for MANGAID 1-102750
gz3d_1-596726_19_14742136: No matching image for MANGAID 1-596726
gz3d_1-443789_37_14731898: No matching image for MANGAID 1-443789
gz3d_1-111536_127_14715938: No matching image for MANGAID 1-111536
gz3d_1-70219_127_14713377: No matching image for MANGAID 1-70219
gz3d_1-117639_127_14716136: No matching image for MANGAID 1-117639
gz3d_1-114923_91_14716050: No matching image for MANGAID 1-114923
gz3d_1-585110_61_14741160: No matching image for MANGAID 1-585110
gz3d_1-563951_37_14740131: No matching image for MANGAID 1-563951
gz3d_1-195230_127_14719496: No matching image for MANGAID 1-195230
gz3d_1-591475_61_14741656: No matching image for MANGAID 1-591475
gz3d_1-103303_37_14715613: No matching image for MANGAID 1-103303
gz3d_1-443801_37_14731900: No matching image for MANGAID 1-443801
gz3d_1-444314_127_14731951: No matching image for MANGAID 1-444314
gz3d_1-212493_91_14720398: No matching image for MANGAID 1-212493
gz3d_1-51727_91_14712038: No matching image for MANGAID 1-51727
gz3d_1-103901_61_14715650: No matching image for MANGAID 1-103901
gz3d_1-50094_91_14711853: No matching image for MANGAID 1-50094
gz3d_1-498133_61_14736075: No matching image for MANGAID 1-498133
gz3d_1-589152_61_14741455: No matching image for MANGAID 1-589152
gz3d_1-51430_127_14712007: No matching image for MANGAID 1-51430
gz3d_1-52284_127_14712081: No matching image for MANGAID 1-52284
gz3d_1-592049_37_14741749: No matching image for MANGAID 1-592049
gz3d_1-105821_91_14715765: No matching image for MANGAID 1-105821
gz3d_1-100779_37_14715446: No matching image for MANGAID 1-100779
gz3d_1-112857_61_14715986: No matching image for MANGAID 1-112857
gz3d_1-42763_61_14711408: No matching image for MANGAID 1-42763
gz3d_1-39376_127_14711189: No matching image for MANGAID 1-39376
gz3d_1-42480_127_14711382: No matching image for MANGAID 1-42480
gz3d_1-580616_127_14741064: No matching image for MANGAID 1-580616
gz3d_1-42199_127_14711354: No matching image for MANGAID 1-42199
gz3d_1-484989_127_14734749: No matching image for MANGAID 1-484989
gz3d_1-50622_37_14711916: No matching image for MANGAID 1-50622
gz3d_1-443163_37_14731787: No matching image for MANGAID 1-443163
gz3d_1-584831_127_14741138: No matching image for MANGAID 1-584831
gz3d_1-521242_61_14737807: No matching image for MANGAID 1-521242
gz3d_1-582638_127_14741093: No matching image for MANGAID 1-582638
gz3d_1-604003_37_14742257: No matching image for MANGAID 1-604003
gz3d_1-482551_61_14734508: No matching image for MANGAID 1-482551
gz3d_1-40745_127_14711264: No matching image for MANGAID 1-40745
gz3d_1-42602_127_14711388: No matching image for MANGAID 1-42602
gz3d_1-421926_127_14730829: No matching image for MANGAID 1-421926
gz3d_1-601866_61_14742189: No matching image for MANGAID 1-601866
gz3d_1-212516_61_14720404: No matching image for MANGAID 1-212516
gz3d_1-310328_127_14724750: No matching image for MANGAID 1-310328
gz3d_1-104636_61_14715694: No matching image for MANGAID 1-104636
gz3d_1-41092_127_14711290: No matching image for MANGAID 1-41092
gz3d_1-118794_37_14716227: No matching image for MANGAID 1-118794
gz3d_1-100349_127_14715427: No matching image for MANGAID 1-100349
gz3d_1-570556_19_14740742: No matching image for MANGAID 1-570556
gz3d_1-51668_61_14712032: No matching image for MANGAID 1-51668
gz3d_1-212392_61_14720377: No matching image for MANGAID 1-212392
gz3d_1-555212_91_14739693: No matching image for MANGAID 1-555212
gz3d_1-117730_127_14716144: No matching image for MANGAID 1-117730
gz3d_1-97184_37_14715322: No matching image for MANGAID 1-97184
gz3d_1-118478_37_14716215: No matching image for MANGAID 1-118478
gz3d_1-110809_37_14715916: No matching image for MANGAID 1-110809
gz3d_1-114818_19_14716044: No matching image for MANGAID 1-114818
gz3d_1-100227_37_14715423: No matching image for MANGAID 1-100227
gz3d_1-428190_127_14731146: No matching image for MANGAID 1-428190
gz3d_1-118940_37_14716232: No matching image for MANGAID 1-118940
gz3d_1-463781_61_14733478: No matching image for MANGAID 1-463781
gz3d_1-523624_61_14737947: No matching image for MANGAID 1-523624
gz3d_1-74796_127_14713789: No matching image for MANGAID 1-74796
gz3d_1-588283_91_14741397: No matching image for MANGAID 1-588283
gz3d_1-101970_61_14715524: No matching image for MANGAID 1-101970
gz3d_1-105663_37_14715754: No matching image for MANGAID 1-105663
gz3d_1-117635_127_14716135: No matching image for MANGAID 1-117635
gz3d_1-456389_127_14732840: No matching image for MANGAID 1-456389
gz3d_1-402198_91_14729593: No matching image for MANGAID 1-402198
gz3d_1-558675_61_14739867: No matching image for MANGAID 1-558675
gz3d_1-593521_91_14741895: No matching image for MANGAID 1-593521
gz3d_1-200987_61_14719885: No matching image for MANGAID 1-200987
gz3d_1-590771_19_14741584: No matching image for MANGAID 1-590771
gz3d_1-585725_61_14741280: No matching image for MANGAID 1-585725
gz3d_1-601195_61_14742175: No matching image for MANGAID 1-601195
gz3d_1-209552_61_14720098: No matching image for MANGAID 1-209552
gz3d_1-318807_37_14725170: No matching image for MANGAID 1-318807
gz3d_1-341255_37_14726419: No matching image for MANGAID 1-341255
gz3d_1-105191_37_14715737: No matching image for MANGAID 1-105191
gz3d_1-591528_61_14741662: No matching image for MANGAID 1-591528
gz3d_1-295128_91_14723920: No matching image for MANGAID 1-295128
gz3d_1-584805_91_14741133: No matching image for MANGAID 1-584805
gz3d_1-455637_61_14732741: No matching image for MANGAID 1-455637
gz3d_1-507534_37_14736791: No matching image for MANGAID 1-507534
gz3d_1-50136_127_14711859: No matching image for MANGAID 1-50136
gz3d_1-608213_61_14742786: No matching image for MANGAID 1-608213
gz3d_1-364880_127_14727884: No matching image for MANGAID 1-364880
gz3d_1-154041_19_14717919: No matching image for MANGAID 1-154041
gz3d_1-577940_127_14741043: No matching image for MANGAID 1-577940
gz3d_1-49756_127_14711818: No matching image for MANGAID 1-49756
gz3d_1-535111_91_14738723: No matching image for MANGAID 1-535111
gz3d_1-212158_19_14720359: No matching image for MANGAID 1-212158
gz3d_1-101911_61_14715520: No matching image for MANGAID 1-101911
gz3d_1-360561_37_14727526: No matching image for MANGAID 1-360561
gz3d_1-98769_61_14715371: No matching image for MANGAID 1-98769
gz3d_1-101346_61_14715471: No matching image for MANGAID 1-101346
gz3d_1-105874_37_14715767: No matching image for MANGAID 1-105874
gz3d_1-396966_37_14729312: No matching image for MANGAID 1-396966
gz3d_1-104258_91_14715669: No matching image for MANGAID 1-104258
gz3d_1-118456_127_14716212: No matching image for MANGAID 1-118456
gz3d_1-51634_19_14712029: No matching image for MANGAID 1-51634
gz3d_1-49903_127_14711830: No matching image for MANGAID 1-49903
gz3d_1-582428_19_14741085: No matching image for MANGAID 1-582428
gz3d_1-592838_127_14741826: No matching image for MANGAID 1-592838
gz3d_1-42273_37_14711363: No matching image for MANGAID 1-42273
gz3d_1-101842_37_14715512: No matching image for MANGAID 1-101842
gz3d_1-221714_127_14720780: No matching image for MANGAID 1-221714
gz3d_1-523810_91_14737963: No matching image for MANGAID 1-523810
gz3d_1-575847_127_14740910: No matching image for MANGAID 1-575847
gz3d_1-118428_37_14716207: No matching image for MANGAID 1-118428
gz3d_1-40810_19_14711274: No matching image for MANGAID 1-40810
gz3d_1-99548_91_14715393: No matching image for MANGAID 1-99548
gz3d_1-638544_127_14744362: No matching image for MANGAID 1-638544
gz3d_1-41141_91_14711293: No matching image for MANGAID 1-41141
gz3d_1-635851_37_14744294: No matching image for MANGAID 1-635851
gz3d_1-101281_61_14715466: No matching image for MANGAID 1-101281
gz3d_1-101341_127_14715470: No matching image for MANGAID 1-101341
gz3d_1-242390_19_14721713: No matching image for MANGAID 1-242390
gz3d_1-308670_127_14724643: No matching image for MANGAID 1-308670
gz3d_1-42867_37_14711414: No matching image for MANGAID 1-42867
gz3d_1-42718_61_14711403: No matching image for MANGAID 1-42718
gz3d_1-542111_127_14739147: No matching image for MANGAID 1-542111
gz3d_1-381172_19_14728615: No matching image for MANGAID 1-381172
gz3d_1-584667_61_14741117: No matching image for MANGAID 1-584667
gz3d_1-443774_37_14731896: No matching image for MANGAID 1-443774
gz3d_1-506996_37_14736744: No matching image for MANGAID 1-506996
gz3d_1-609412_127_14742834: No matching image for MANGAID 1-609412
gz3d_1-51767_127_14712051: No matching image for MANGAID 1-51767
gz3d_1-590587_61_14741581: No matching image for MANGAID 1-590587
gz3d_1-212898_61_14720425: No matching image for MANGAID 1-212898
gz3d_1-67776_91_14713121: No matching image for MANGAID 1-67776
gz3d_1-581227_127_14741076: No matching image for MANGAID 1-581227
gz3d_1-100495_19_14715432: No matching image for MANGAID 1-100495
gz3d_1-41160_127_14711295: No matching image for MANGAID 1-41160
gz3d_1-96434_37_14715288: No matching image for MANGAID 1-96434
gz3d_1-168792_37_14718600: No matching image for MANGAID 1-168792
gz3d_1-118184_61_14716178: No matching image for MANGAID 1-118184
gz3d_1-105140_37_14715734: No matching image for MANGAID 1-105140
gz3d_1-101769_61_14715505: No matching image for MANGAID 1-101769
gz3d_1-639152_19_14744456: No matching image for MANGAID 1-639152
gz3d_1-40777_37_14711268: No matching image for MANGAID 1-40777
gz3d_1-404371_61_14729792: No matching image for MANGAID 1-404371
gz3d_1-585593_37_14741243: No matching image for MANGAID 1-585593
gz3d_1-541323_127_14739101: No matching image for MANGAID 1-541323
gz3d_1-456917_91_14733060: No matching image for MANGAID 1-456917
gz3d_1-639541_127_14744465: No matching image for MANGAID 1-639541
gz3d_1-286805_127_14723821: No matching image for MANGAID 1-286805
gz3d_1-101235_37_14715464: No matching image for MANGAID 1-101235
gz3d_1-51939_127_14712069: No matching image for MANGAID 1-51939
gz3d_1-105767_127_14715760: No matching image for MANGAID 1-105767
gz3d_1-40883_19_14711281: No matching image for MANGAID 1-40883
gz3d_1-453021_61_14732539: No matching image for MANGAID 1-453021
gz3d_1-425496_127_14731032: No matching image for MANGAID 1-425496
gz3d_1-635526_61_14744279: No matching image for MANGAID 1-635526
gz3d_1-36522_127_14710945: No matching image for MANGAID 1-36522
gz3d_1-100139_61_14715419: No matching image for MANGAID 1-100139
gz3d_1-240325_127_14721618: No matching image for MANGAID 1-240325
gz3d_1-333696_127_14725956: No matching image for MANGAID 1-333696
gz3d_1-596877_127_14742139: No matching image for MANGAID 1-596877
gz3d_1-593890_127_14741983: No matching image for MANGAID 1-593890
gz3d_1-51631_37_14712028: No matching image for MANGAID 1-51631
gz3d_1-398008_127_14729356: No matching image for MANGAID 1-398008
gz3d_1-101770_127_14715506: No matching image for MANGAID 1-101770
gz3d_1-598278_61_14742148: No matching image for MANGAID 1-598278
gz3d_1-102101_61_14715540: No matching image for MANGAID 1-102101
gz3d_1-335372_19_14726050: No matching image for MANGAID 1-335372
gz3d_1-42587_37_14711386: No matching image for MANGAID 1-42587
gz3d_1-119315_61_14716241: No matching image for MANGAID 1-119315
gz3d_1-99875_91_14715412: No matching image for MANGAID 1-99875
gz3d_1-459682_37_14733296: No matching image for MANGAID 1-459682
gz3d_1-110509_127_14715905: No matching image for MANGAID 1-110509
gz3d_1-570549_127_14740739: No matching image for MANGAID 1-570549
gz3d_1-51369_127_14711997: No matching image for MANGAID 1-51369
gz3d_1-578294_127_14741046: No matching image for MANGAID 1-578294
gz3d_1-117892_19_14716154: No matching image for MANGAID 1-117892
gz3d_1-594067_37_14742012: No matching image for MANGAID 1-594067
gz3d_1-96668_127_14715304: No matching image for MANGAID 1-96668
gz3d_1-100963_91_14715454: No matching image for MANGAID 1-100963
gz3d_1-604068_61_14742271: No matching image for MANGAID 1-604068
gz3d_1-572558_37_14740758: No matching image for MANGAID 1-572558
gz3d_1-112837_19_14715985: No matching image for MANGAID 1-112837
gz3d_1-104203_127_14715668: No matching image for MANGAID 1-104203
gz3d_1-510624_61_14736968: No matching image for MANGAID 1-510624
gz3d_1-51135_37_14711980: No matching image for MANGAID 1-51135
gz3d_1-105089_127_14715729: No matching image for MANGAID 1-105089
gz3d_1-212626_37_14720408: No matching image for MANGAID 1-212626
gz3d_1-577296_19_14741040: No matching image for MANGAID 1-577296
gz3d_1-317589_37_14725079: No matching image for MANGAID 1-317589
gz3d_1-115464_19_14716062: No matching image for MANGAID 1-115464
gz3d_1-97656_37_14715339: No matching image for MANGAID 1-97656
gz3d_1-578386_127_14741053: No matching image for MANGAID 1-578386
gz3d_1-69248_37_14713272: No matching image for MANGAID 1-69248
gz3d_1-101347_127_14715472: No matching image for MANGAID 1-101347
gz3d_1-114292_37_14716030: No matching image for MANGAID 1-114292
gz3d_1-498205_37_14736092: No matching image for MANGAID 1-498205
gz3d_1-591294_127_14741629: No matching image for MANGAID 1-591294
gz3d_1-423296_61_14730917: No matching image for MANGAID 1-423296
gz3d_1-118204_127_14716180: No matching image for MANGAID 1-118204
gz3d_1-593051_127_14741842: No matching image for MANGAID 1-593051
gz3d_1-102602_61_14715573: No matching image for MANGAID 1-102602
gz3d_1-118159_37_14716174: No matching image for MANGAID 1-118159
gz3d_1-105097_127_14715730: No matching image for MANGAID 1-105097
gz3d_1-637399_37_14744318: No matching image for MANGAID 1-637399
gz3d_1-101877_61_14715516: No matching image for MANGAID 1-101877
gz3d_1-42214_19_14711356: No matching image for MANGAID 1-42214
gz3d_1-598275_37_14742147: No matching image for MANGAID 1-598275
gz3d_1-116661_91_14716079: No matching image for MANGAID 1-116661
gz3d_1-441283_91_14731673: No matching image for MANGAID 1-441283
gz3d_1-50901_61_14711945: No matching image for MANGAID 1-50901
gz3d_1-169669_127_14718638: No matching image for MANGAID 1-169669
gz3d_1-113208_37_14715999: No matching image for MANGAID 1-113208
gz3d_1-503114_127_14736400: No matching image for MANGAID 1-503114
gz3d_1-41843_61_14711332: No matching image for MANGAID 1-41843
gz3d_1-298366_91_14724068: No matching image for MANGAID 1-298366
gz3d_1-587846_127_14741373: No matching image for MANGAID 1-587846
gz3d_1-117419_127_14716123: No matching image for MANGAID 1-117419
gz3d_1-555204_37_14739690: No matching image for MANGAID 1-555204
gz3d_1-447868_127_14732175: No matching image for MANGAID 1-447868
gz3d_1-116518_61_14716074: No matching image for MANGAID 1-116518
gz3d_1-118223_127_14716184: No matching image for MANGAID 1-118223
gz3d_1-115375_19_14716060: No matching image for MANGAID 1-115375
gz3d_1-539653_127_14738997: No matching image for MANGAID 1-539653
gz3d_1-631500_127_14743941: No matching image for MANGAID 1-631500
gz3d_1-321149_61_14725293: No matching image for MANGAID 1-321149
gz3d_1-68636_127_14713185: No matching image for MANGAID 1-68636
gz3d_1-541296_37_14739098: No matching image for MANGAID 1-541296
gz3d_1-117658_91_14716137: No matching image for MANGAID 1-117658
gz3d_1-117984_19_14716164: No matching image for MANGAID 1-117984
gz3d_1-558772_127_14739869: No matching image for MANGAID 1-558772
gz3d_1-541232_127_14739094: No matching image for MANGAID 1-541232
gz3d_1-474362_127_14734123: No matching image for MANGAID 1-474362
gz3d_1-570686_37_14740751: No matching image for MANGAID 1-570686
gz3d_1-96245_19_14715274: No matching image for MANGAID 1-96245
gz3d_1-638039_127_14744324: No matching image for MANGAID 1-638039
gz3d_1-52357_19_14712082: No matching image for MANGAID 1-52357
gz3d_1-389362_127_14729021: No matching image for MANGAID 1-389362
gz3d_1-456794_91_14733002: No matching image for MANGAID 1-456794
gz3d_1-117122_127_14716107: No matching image for MANGAID 1-117122
gz3d_1-49841_127_14711827: No matching image for MANGAID 1-49841
gz3d_1-71124_127_14713452: No matching image for MANGAID 1-71124
gz3d_1-317060_61_14725055: No matching image for MANGAID 1-317060
gz3d_1-80331_37_14714279: No matching image for MANGAID 1-80331
gz3d_1-109744_61_14715866: No matching image for MANGAID 1-109744
gz3d_1-117610_61_14716134: No matching image for MANGAID 1-117610
gz3d_1-244489_127_14721783: No matching image for MANGAID 1-244489
gz3d_1-436238_91_14731456: No matching image for MANGAID 1-436238
gz3d_1-51397_91_14712002: No matching image for MANGAID 1-51397
gz3d_1-351890_127_14726989: No matching image for MANGAID 1-351890
gz3d_1-50537_61_14711909: No matching image for MANGAID 1-50537
gz3d_1-117694_19_14716140: No matching image for MANGAID 1-117694
gz3d_1-39880_91_14711216: No matching image for MANGAID 1-39880
gz3d_1-624146_127_14743552: No matching image for MANGAID 1-624146
gz3d_1-112576_127_14715974: No matching image for MANGAID 1-112576
gz3d_1-51326_61_14711989: No matching image for MANGAID 1-51326
gz3d_1-104810_61_14715705: No matching image for MANGAID 1-104810
gz3d_1-403615_127_14729670: No matching image for MANGAID 1-403615
gz3d_1-39741_61_14711205: No matching image for MANGAID 1-39741
gz3d_1-69640_127_14713315: No matching image for MANGAID 1-69640
gz3d_1-318435_61_14725131: No matching image for MANGAID 1-318435
gz3d_1-52494_37_14712089: No matching image for MANGAID 1-52494
gz3d_1-457168_61_14733137: No matching image for MANGAID 1-457168
gz3d_1-380334_19_14728570: No matching image for MANGAID 1-380334
gz3d_1-96265_37_14715276: No matching image for MANGAID 1-96265
gz3d_1-98819_37_14715378: No matching image for MANGAID 1-98819
gz3d_1-591263_91_14741622: No matching image for MANGAID 1-591263
gz3d_1-329001_19_14725674: No matching image for MANGAID 1-329001
gz3d_1-594386_19_14742072: No matching image for MANGAID 1-594386
gz3d_1-577986_127_14741045: No matching image for MANGAID 1-577986
gz3d_1-586360_127_14741301: No matching image for MANGAID 1-586360
gz3d_1-41304_127_14711302: No matching image for MANGAID 1-41304
gz3d_1-460580_37_14733329: No matching image for MANGAID 1-460580
gz3d_1-105764_61_14715759: No matching image for MANGAID 1-105764
gz3d_1-42202_37_14711355: No matching image for MANGAID 1-42202
gz3d_1-104866_127_14715712: No matching image for MANGAID 1-104866
gz3d_1-39805_37_14711211: No matching image for MANGAID 1-39805
gz3d_1-456902_127_14733047: No matching image for MANGAID 1-456902
gz3d_1-97454_37_14715328: No matching image for MANGAID 1-97454
gz3d_1-109872_37_14715874: No matching image for MANGAID 1-109872
gz3d_1-594349_37_14742070: No matching image for MANGAID 1-594349
gz3d_1-603028_61_14742230: No matching image for MANGAID 1-603028
gz3d_1-110331_37_14715896: No matching image for MANGAID 1-110331
gz3d_1-210098_127_14720130: No matching image for MANGAID 1-210098
gz3d_1-612570_127_14742940: No matching image for MANGAID 1-612570
gz3d_1-120061_37_14716261: No matching image for MANGAID 1-120061
gz3d_1-104455_37_14715679: No matching image for MANGAID 1-104455
gz3d_1-587343_61_14741349: No matching image for MANGAID 1-587343
gz3d_1-101290_91_14715467: No matching image for MANGAID 1-101290
gz3d_1-400001_127_14729481: No matching image for MANGAID 1-400001
gz3d_1-117799_37_14716150: No matching image for MANGAID 1-117799
gz3d_1-124128_61_14716454: No matching image for MANGAID 1-124128
gz3d_1-473096_127_14734048: No matching image for MANGAID 1-473096
gz3d_1-602993_127_14742229: No matching image for MANGAID 1-602993
gz3d_1-284552_37_14723724: No matching image for MANGAID 1-284552
gz3d_1-415871_19_14730376: No matching image for MANGAID 1-415871
gz3d_1-51113_61_14711972: No matching image for MANGAID 1-51113
gz3d_1-359453_127_14727420: No matching image for MANGAID 1-359453
gz3d_1-98209_91_14715354: No matching image for MANGAID 1-98209
gz3d_1-112338_127_14715966: No matching image for MANGAID 1-112338
gz3d_1-117936_61_14716156: No matching image for MANGAID 1-117936
gz3d_1-456788_37_14733001: No matching image for MANGAID 1-456788
gz3d_1-593702_127_14741934: No matching image for MANGAID 1-593702
gz3d_1-115630_127_14716064: No matching image for MANGAID 1-115630
gz3d_1-51165_37_14711983: No matching image for MANGAID 1-51165
gz3d_1-591083_127_14741596: No matching image for MANGAID 1-591083
gz3d_1-41350_91_14711304: No matching image for MANGAID 1-41350
gz3d_1-580957_127_14741073: No matching image for MANGAID 1-580957
gz3d_1-605125_127_14742376: No matching image for MANGAID 1-605125
gz3d_1-628673_127_14743797: No matching image for MANGAID 1-628673
gz3d_1-578364_37_14741050: No matching image for MANGAID 1-578364
gz3d_1-601046_91_14742168: No matching image for MANGAID 1-601046
gz3d_1-582721_61_14741094: No matching image for MANGAID 1-582721
gz3d_1-51793_127_14712056: No matching image for MANGAID 1-51793
gz3d_1-117978_91_14716162: No matching image for MANGAID 1-117978
gz3d_1-50918_37_14711948: No matching image for MANGAID 1-50918
gz3d_1-280830_127_14723494: No matching image for MANGAID 1-280830
gz3d_1-352066_127_14726999: No matching image for MANGAID 1-352066
gz3d_1-315503_127_14724989: No matching image for MANGAID 1-315503
gz3d_1-102801_127_14715581: No matching image for MANGAID 1-102801
gz3d_1-103520_19_14715627: No matching image for MANGAID 1-103520
gz3d_1-364267_37_14727831: No matching image for MANGAID 1-364267
gz3d_1-627182_37_14743733: No matching image for MANGAID 1-627182
"""

# --- Helper: extract MANGAIDs from log ---
import re
MANGAIDS = sorted(set(re.findall(r'MANGAID (\d+-\d+)', SKIPPED_LOG)))

# --- Load metadata ---
meta_df = pd.read_csv(META_CSV)
meta_df['MANGAID'] = meta_df['MANGAID'].astype(str)
meta_dict = meta_df.set_index('MANGAID')[['ra', 'dec']].to_dict('index')

# --- Processing parameters ---
clipLimit_range = np.linspace(1.0, 5.0, 10)
tileGridSize_range = [(i, i) for i in range(3, 8, 1)]
MAX_VARIANTS = 50

def download_legacy_fits(ra, dec, bands='grz', outfn='cutout.fits'):
    layer = 'ls-dr10'
    pixscale = 0.099
    size_px = 525
    url = (f'https://www.legacysurvey.org/viewer/cutout.fits'
           f'?ra={ra}&dec={dec}&layer={layer}&pixscale={pixscale}&size={size_px}&bands={bands}')
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(outfn, 'wb') as f:
        for chunk in r.iter_content(1024*32):
            f.write(chunk)

def safe_normalize(arr):
    arr = np.nan_to_num(arr)
    arr -= np.min(arr)
    max_val = np.max(arr)
    return arr / max_val if max_val > 0 else np.zeros_like(arr)

def enhance_bar(arr, clipLimit, tileGridSize):
    arr = safe_normalize(arr)
    arr_uint8 = (arr * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(arr_uint8)

def compute_bisymmetric_strength(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    height, width = arr.shape
    cy, cx = height // 2, width // 2
    max_r = min(cx, cy, 100)
    dr = 2
    r_bins = np.arange(dr, max_r, step=dr)
    a2_values = []

    yy, xx = np.meshgrid(np.arange(height), np.arange(width))
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    thetas = np.arctan2(dy, dx)

    for r in r_bins:
        mask = (dist >= r - dr/2) & (dist < r + dr/2)
        if np.sum(mask) < 10:
            continue
        intensities = arr[mask]
        theta_bin = thetas[mask]
        total_intensity = np.sum(intensities)
        if total_intensity == 0:
            continue
        cos_sum = np.sum(intensities * np.cos(2 * theta_bin))
        sin_sum = np.sum(intensities * np.sin(2 * theta_bin))
        a2 = np.sqrt(cos_sum**2 + sin_sum**2) / total_intensity
        a2_values.append(a2)

    return np.max(a2_values) if a2_values else 0.0

def processed_image_exists(mangaid, ra, dec):
    base_name = f"{mangaid}_zoomed_ra{ra}_dec{dec}"
    for fname in os.listdir(BEST_DIR):
        if fname.startswith(base_name) and fname.endswith('.png'):
            return True
    return False

def fits_exists(mangaid, ra, dec):
    fits_name = f"{mangaid}_zoomed_ra{ra}_dec{dec}.fits"
    return os.path.exists(os.path.join(OUT_FITS_DIR, fits_name))

def download_one(mangaid):
    if mangaid not in meta_dict:
        return (mangaid, False, "No metadata")
    ra, dec = meta_dict[mangaid]['ra'], meta_dict[mangaid]['dec']
    fits_name = f"{mangaid}_zoomed_ra{ra}_dec{dec}.fits"
    fits_path = os.path.join(OUT_FITS_DIR, fits_name)
    if os.path.exists(fits_path):
        return (mangaid, True, "Already downloaded")
    try:
        download_legacy_fits(ra, dec, outfn=fits_path)
        return (mangaid, True, "Downloaded")
    except Exception as e:
        return (mangaid, False, str(e))

# --- Parallel download with progress bar ---
to_download = []
for mangaid in MANGAIDS:
    if mangaid not in meta_dict:
        continue
    ra, dec = meta_dict[mangaid]['ra'], meta_dict[mangaid]['dec']
    if not processed_image_exists(mangaid, ra, dec):
        to_download.append(mangaid)

if to_download:
    print(f"Downloading {len(to_download)} FITS files in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, mangaid): mangaid for mangaid in to_download}
        for f in tqdm(as_completed(futures), total=len(futures), desc="FITS Download"):
            pass

# --- Process FITS files with progress bar, skip already processed ---
for mangaid in tqdm(MANGAIDS, desc="Processing", unit="galaxy"):
    if mangaid not in meta_dict:
        continue
    ra, dec = meta_dict[mangaid]['ra'], meta_dict[mangaid]['dec']
    if processed_image_exists(mangaid, ra, dec):
        continue
    fits_name = f"{mangaid}_zoomed_ra{ra}_dec{dec}.fits"
    fits_path = os.path.join(OUT_FITS_DIR, fits_name)
    if not os.path.exists(fits_path):
        continue

    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None or data.ndim < 3 or data.shape[0] < 3:
                continue
            g_band, r_band, z_band = data[0], data[1], data[2]
            combined_data = (g_band + r_band + z_band) / 3.0
            base_name = os.path.splitext(os.path.basename(fits_path))[0]

            best_strength = 0.0
            best_img = None
            best_info = {"clip": None, "tile": None}

            count = 0
            for clipLimit in clipLimit_range:
                for tileGridSize in tileGridSize_range:
                    if count >= MAX_VARIANTS:
                        break
                    bw_enhanced = enhance_bar(combined_data, clipLimit, tileGridSize)
                    temp_img = Image.fromarray(bw_enhanced)
                    temp_path = f"/tmp/{base_name}_temp.png"
                    temp_img.save(temp_path)
                    strength = compute_bisymmetric_strength(temp_path)
                    os.remove(temp_path)
                    if strength > best_strength:
                        best_strength = strength
                        best_img = bw_enhanced.copy()
                        best_info = {"clip": clipLimit, "tile": tileGridSize}
                    count += 1
                if count >= MAX_VARIANTS:
                    break

            if best_img is not None:
                best_name = f"{base_name}__BEST__clip_{best_info['clip']:.1f}__tile_{best_info['tile'][0]}x{best_info['tile'][1]}.png"
                Image.fromarray(best_img).save(os.path.join(BEST_DIR, best_name))
    except Exception:
        continue

import json, matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
def plot_all_cms(dpi=72):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(allcms)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh), dpi=dpi)
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)
    axs[0].set_title('Custom colormaps', fontsize=14)

    for ax, cmap in zip(axs, allcms):
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.text(-0.01, 0.5, cmap.name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

allcms = []

_mycool = json.loads(["#00305d","#002e59","#002c55","#002b51","#00294e","#00284a","#002748","#002645","#002543","#002541","#00253f","#00243d","#00243c","#00243b","#00243a","#00243a","#082539","#102539","#172639","#1f263a","#25273a","#2c283b","#32293c","#382a3d","#3d2b3f","#422c40","#472d42","#4c2e44","#503046","#543148","#58324b","#5c344d","#603550","#633753","#663856","#693a5a","#6c3c5d","#6f3d60","#723f64","#744068","#77426c","#794470","#7b4674","#7e4778","#80497c","#824b81","#844c85","#864e8a","#88508f","#8a5294","#8c5399","#8e559e","#9057a3","#9258a8","#945aad","#965cb2","#985db8","#9a5fbd","#9c61c2","#9f62c8","#a164cd","#a366d3","#a567d8","#a769de","#aa6ae4","#ac6ce9","#af6def","#b16ff5","#b470fa","#b672ff","#b973ff","#bc75ff","#be76ff","#c177ff","#c479ff","#c77aff","#c97cff","#cc7dff","#cf7eff","#d280ff","#d581ff","#d882ff","#db84ff","#de85ff","#e187ff","#e488ff","#e789ff","#ea8bff","#ed8cff","#f08eff","#f38fff","#f591ff","#f892ff","#fb94ff","#fd95ff","#ff97ff","#ff98ff","#ff9aff","#ff9cff","#ff9eff","#ff9fff","#ffa1ff","#ffa3ff","#ffa5ff","#ffa7ff","#ffaaff","#ffacff","#ffaeff","#ffb1ff","#ffb3ff","#ffb6ff","#ffb8ff","#ffbbff","#ffbeff","#ffc1ff","#ffc4ff","#ffc8ff","#ffcbff","#ffcfff","#fed3ff","#fad6ff","#f5daff","#f0dfff","#eae3ff","#e4e8ff","#ddecff","#d6f1ff","#cdf6ff"])
mycool = LinearSegmentedColormap.from_list('mycool',_mycool)
allcms.append(mycool)

### From Rachel ###
_myspringcs = json.loads('["#000040","#000041","#070041","#0e0042","#150043","#1c0044","#230044","#290045","#300146","#370247","#3d0347","#440448","#4a0449","#50054a","#56064a","#5c074b","#62084c","#68094c","#6e0a4d","#740b4e","#790c4f","#7f0d4f","#840e50","#8a0f51","#8f1152","#941252","#991353","#9e1454","#a31655","#a81755","#ad1856","#b11957","#b61b58","#bb1c58","#bf1e59","#c31f5a","#c7215b","#cc225b","#d0245c","#d4255d","#d7275e","#db285e","#df2a5f","#e22c60","#e62d61","#e92f61","#ed3162","#f03363","#f33464","#f63664","#f93865","#fc3a66","#ff3c67","#ff3e67","#ff4068","#ff4269","#ff446a","#ff466a","#ff486b","#ff4a6c","#ff4c6d","#ff4e6d","#ff506e","#ff536f","#ff5570","#ff5770","#ff5971","#ff5c72","#ff5e73","#ff6073","#ff6374","#ff6575","#ff6776","#ff6a76","#ff6c77","#ff6f78","#ff7179","#ff7479","#ff777a","#ff797b","#ff7c7c","#ff7f7c","#ff817d","#ff847e","#ff877f","#ff897f","#ff8c80","#ff8f81","#ff9282","#ff9582","#ff9883","#ff9b84","#ff9e85","#ffa185","#ffa486","#ffa787","#ffaa88","#ffad88","#ffb089","#ffb38a","#ffb68b","#ffb98b","#ffbd8c","#ffc08d","#ffc38e","#ffc68f","#ffca8f","#ffcd90","#ffd091","#ffd492","#ffd792","#ffdb93","#ffde94","#ffe295","#ffe595","#fce996","#f9ec97","#f6f098","#f3f498","#f0f799","#edfb9a","#e9ff9b","#e6ff9b","#e2ff9c","#dfff9d","#dbff9e","#d7ff9f","#d4ff9f"]')
myspringcs = LinearSegmentedColormap.from_list('myspringcs',_myspringcs)
allcms.append(myspringcs)

_myfunnyvalentinecs = json.loads('["#410400","#450600","#490700","#4d0800","#510900","#550a00","#590b00","#5d0c00","#610d00","#650e00","#680f00","#6c1000","#701200","#741300","#781400","#7c1500","#7f1600","#831700","#871900","#8a1a00","#8e1b00","#921c00","#961d00","#991f00","#9d2000","#a02100","#a42200","#a82300","#ab2500","#af2600","#b22700","#b62800","#b92a00","#bd2b00","#c02c00","#c42e00","#c72f00","#ca3000","#ce3100","#d13300","#d43400","#d83500","#db3700","#de3800","#e23900","#e53b00","#e83c00","#eb3e04","#ef3f09","#f2400d","#f54212","#f84317","#fb451b","#fe4620","#ff4725","#ff4929","#ff4a2e","#ff4c33","#ff4d37","#ff4f3c","#ff5040","#ff5145","#ff5349","#ff544e","#ff5652","#ff5757","#ff595b","#ff5a60","#ff5c64","#ff5d69","#ff5f6d","#ff6071","#ff6276","#ff647a","#ff657f","#ff6783","#ff6887","#ff6a8b","#ff6b90","#ff6d94","#ff6f98","#ff709d","#ff72a1","#ff73a5","#ff75a9","#ff77ad","#ff78b2","#ff7ab6","#ff7cba","#ff7dbe","#ff7fc2","#ff81c6","#ff82ca","#ff84ce","#ff86d2","#ff87d6","#ff89da","#ff8bde","#ff8ce2","#ff8ee6","#ff90ea","#ff92ee","#ff93f2","#ff95f6","#ff97fa","#ff99fe","#ff9aff","#ff9cff","#ff9eff","#ffa0ff","#ffa1ff","#ffa3ff","#ffa5ff","#ffa7ff","#ffa9ff","#ffabff","#ffacff","#ffaeff","#ffb0ff","#ffb2ff","#ffb4ff","#ffb6ff","#ffb8ff","#ffb9ff","#ffbbff","#ffbdff","#ffbfff","#ffc1ff"]')
myfunnyvalentinecs = LinearSegmentedColormap.from_list('myfunnyvalentinecs',_myfunnyvalentinecs)
allcms.append(myfunnyvalentinecs)

_mymosscs = json.loads('["#001f0f","#00200f","#00210f","#00230f","#002410","#002510","#002710","#002810","#002910","#002b10","#002c10","#002d10","#002f11","#003011","#003211","#003311","#003511","#003611","#003811","#003912","#003b12","#003c12","#003e12","#003f12","#034112","#074212","#0b4412","#0f4612","#134713","#174913","#1b4a13","#1e4c13","#224e13","#264f13","#2a5113","#2d5313","#315513","#345614","#385814","#3b5a14","#3f5c14","#425d14","#455f14","#496114","#4c6314","#4f6514","#526714","#556915","#586a15","#5b6c15","#5e6e15","#617015","#647215","#677415","#6a7615","#6c7815","#6f7a15","#727c15","#747e15","#778016","#798216","#7c8416","#7e8616","#818816","#838b16","#858d16","#888f16","#8a9116","#8c9316","#8e9516","#909716","#929a16","#949c16","#969e17","#98a017","#9aa317","#9ba517","#9da717","#9fa917","#a0ac17","#a2ae17","#a4b017","#a5b317","#a7b517","#a8b817","#a9ba17","#abbc17","#acbf17","#adc117","#aec417","#b0c617","#b1c918","#b2cb18","#b3ce18","#b4d018","#b5d318","#b5d518","#b6d818","#b7da18","#b8dd18","#b9df18","#b9e218","#bae518","#bae718","#bbea18","#bbed18","#bcef18","#bcf218","#bcf518","#bdf718","#bdfa18","#bdfd18","#bdff18","#bdff18","#beff18","#beff18","#beff18","#bdff18","#bdff18","#bdff18","#bdff18","#bdff18","#bcff18","#bcff18","#bcff18","#bbff18","#bbff18","#baff18"]')
mymosscs = LinearSegmentedColormap.from_list('mymosscs',_mymosscs)
allcms.append(mymosscs)

_myterrecs = json.loads('["#002300","#002400","#002500","#002600","#002700","#002800","#002900","#002a00","#002b00","#002c00","#002d00","#002e00","#002f00","#003100","#003200","#003300","#003400","#003500","#003600","#043700","#0a3800","#0f3a00","#143b00","#193c00","#1f3d00","#243f00","#294000","#2e4100","#324200","#374400","#3c4500","#404600","#454800","#4a4900","#4e4a00","#524c00","#574d00","#5b4e00","#5f5000","#635100","#675300","#6b5400","#6f5500","#725700","#765800","#7a5a02","#7d5b05","#815d07","#845e0a","#88600d","#8b6210","#8e6312","#916515","#946619","#97681c","#9a691f","#9d6b22","#a06d26","#a36e29","#a5702d","#a87230","#aa7334","#ad7538","#af773c","#b17940","#b37a44","#b67c48","#b87e4d","#ba8051","#bc8155","#bd835a","#bf855f","#c18763","#c38968","#c48a6d","#c68c72","#c78e77","#c8907c","#ca9281","#cb9487","#cc968c","#cd9891","#ce9a97","#cf9c9d","#d09ea2","#d1a0a8","#d1a2ae","#d2a4b4","#d3a6ba","#d3a8c0","#d3aac7","#d4accd","#d4aed3","#d4b0da","#d4b2e0","#d5b4e7","#d5b6ee","#d4b8f5","#d4bbfc","#d4bdff","#d4bfff","#d3c1ff","#d3c3ff","#d3c6ff","#d2c8ff","#d1caff","#d1ccff","#d0cfff","#cfd1ff","#ced3ff","#cdd5ff","#ccd8ff","#cbdaff","#cadcff","#c8dfff","#c7e1ff","#c6e3ff","#c4e6ff","#c2e8ff","#c1ebff","#bfedff","#bdf0ff","#bbf2ff","#baf4ff","#b8f7ff","#b6f9ff","#b3fcff","#b1feff"]')
myterrecs = LinearSegmentedColormap.from_list('myterrecs',_myterrecs)
allcms.append(myterrecs)

_myprettycs = json.loads('["#001b00","#001a00","#001900","#001800","#001700","#00170b","#001633","#001558","#00147a","#001499","#0013b6","#0013d1","#0012e9","#0012ff","#0012ff","#0011ff","#0011ff","#0011ff","#0011ff","#0411ff","#0a11ff","#1012ff","#1712ff","#1d13ff","#2413ff","#2a14ff","#3115ff","#3816ff","#3f17ff","#4518ff","#4c19ff","#531aff","#591cff","#5f1dff","#651fff","#6b21ff","#7123ff","#7725ff","#7c27ff","#8129ff","#862cff","#8b2eff","#8f31ff","#9334fd","#9736f2","#9a39e6","#9d3cdb","#9f40cf","#a143c4","#a346b9","#a54aad","#a64ea2","#a65197","#a7558c","#a75982","#a65d77","#a5616d","#a46563","#a26a5a","#a06e51","#9d7248","#9a7740","#977b38","#938031","#8f852a","#8b8a23","#868e1e","#819318","#7c9813","#769d0f","#70a20b","#6aa708","#63ac06","#5cb104","#55b603","#4ebb02","#46c002","#3fc503","#37ca04","#2fcf05","#27d408","#1fd90b","#16de0f","#0ee313","#06e818","#00ec1d","#00f123","#00f62a","#00fa31","#00fe39","#00ff41","#00ff4a","#00ff53","#00ff5d","#00ff67","#00ff72","#00ff7d","#00ff88","#00ff94","#00ffa0","#00ffad","#00ffb9","#00ffc6","#00ffd4","#00ffe1","#00ffef","#00fffc","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#06ffff","#19ffff","#2dffff","#43ffff","#5cffff","#76ffff","#92ffff","#b0fdff","#d0f5ff"]')
myprettycs = LinearSegmentedColormap.from_list('myprettycs',_myprettycs)
allcms.append(myprettycs)

_mypretty2cs = json.loads('["#001b00","#002100","#002600","#002b00","#003000","#003400","#003800","#003b00","#003e00","#004100","#004300","#004500","#004700","#004800","#004a00","#004b00","#004c00","#004c00","#004d00","#004d00","#004d00","#024d00","#094d00","#114d00","#184c00","#204c00","#274b00","#2f4b00","#374a00","#3f4900","#474900","#4f4800","#574800","#5f4700","#674600","#6e4600","#764500","#7d4500","#844400","#8b4402","#92440e","#984319","#9e4324","#a4432f","#a9433a","#ae4444","#b3444f","#b7445a","#bb4564","#bf466e","#c24778","#c54882","#c7498c","#c94a95","#cb4b9e","#cc4da7","#cd4fb0","#cd50b8","#cd53c0","#cc55c8","#cb57d0","#ca59d7","#c85cde","#c65fe5","#c362eb","#c065f2","#bd68f7","#b96bfd","#b56fff","#b173ff","#ac76ff","#a67aff","#a17eff","#9b83ff","#9587ff","#8f8bff","#8890ff","#8194ff","#7a99ff","#739eff","#6ba3ff","#63a8ff","#5cadff","#54b2ff","#4cb7ff","#44bcff","#3cc1ff","#34c6ff","#2cccff","#25d1ff","#1dd6ff","#16dbff","#0ee0ff","#07e5ff","#01eaff","#00efff","#00f4ff","#00f9ff","#00fdff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00ffff","#00fffe","#00fffc","#00fffa","#00fff8","#00fff6","#00fff4","#0cfff3","#18fff2","#26fff1","#36fff1","#47fff1","#5afff1","#6ffff2","#86fff3","#9efff4","#b9fff6","#d5fff9","#f4fffc"]')
mypretty2cs = LinearSegmentedColormap.from_list('mypretty2cs',_mypretty2cs)
allcms.append(mypretty2cs)

_mypretty3cs = json.loads('["#001b00","#001a00","#091a00","#121900","#1b1900","#231900","#2b1900","#321900","#391900","#401a00","#461a00","#4c1b00","#521b00","#571c00","#5c1d00","#601e00","#641f00","#682000","#6c2100","#6f2200","#722400","#752500","#772700","#792800","#7b2a00","#7c2c00","#7d2d00","#7e2f00","#7f3100","#7f3300","#803500","#803700","#803900","#7f3c00","#7f3e00","#7e4000","#7d4300","#7c4500","#7a4800","#794a00","#774d00","#754f07","#73520e","#715515","#6f581c","#6d5a23","#6b5d2a","#686031","#656338","#63663f","#606946","#5d6c4d","#5a6f54","#57725a","#547561","#517868","#4e7b6f","#4b7e75","#48817c","#458483","#428789","#3e8a90","#3b8d96","#38909c","#3593a2","#3297a9","#2f9aaf","#2c9db5","#29a0bb","#26a3c0","#24a6c6","#21a9cc","#1eacd1","#1cafd7","#1ab2dc","#17b5e1","#15b8e6","#13bbeb","#11bef0","#0fc1f5","#0ec3f9","#0cc6fe","#0bc9ff","#0accff","#09ceff","#08d1ff","#08d4ff","#07d6ff","#07d9ff","#07dbff","#07ddff","#08e0ff","#09e2ff","#0ae4ff","#0be7ff","#0ce9ff","#0eebff","#10edff","#13efff","#15f1ff","#18f2ff","#1bf4ff","#1ff6ff","#23f7ff","#27f9ff","#2cfaff","#30fcff","#36fdff","#3bfeff","#41ffff","#48ffff","#4effff","#55ffff","#5dffff","#65ffff","#6dffff","#76ffff","#7fffff","#89ffff","#93ffff","#9dffff","#a8ffff","#b4ffff","#c0ffff","#ccffff","#d9ffff","#e6ffff","#f4fffc"]')
mypretty3cs = LinearSegmentedColormap.from_list('mypretty3cs',_mypretty3cs)
allcms.append(mypretty3cs)

_lavhaze = json.loads('["#37002d","#380031","#3a0035","#3b0039","#3d003d","#3f0041","#400044","#420047","#44004b","#46004d","#470050","#490153","#4b0355","#4d0557","#4f0759","#50095b","#520b5d","#540d5e","#560f60","#581161","#5a1462","#5c1663","#5d1864","#5f1a65","#611c65","#631e66","#652166","#672367","#692567","#6b2767","#6d2967","#6f2c67","#712e67","#733067","#753267","#763567","#783766","#7a3966","#7c3b66","#7e3e65","#804065","#824264","#844564","#864763","#884963","#894c62","#8b4e62","#8d5061","#8f5361","#915560","#925760","#94595f","#965c5f","#985e5e","#99605e","#9b635e","#9d655d","#9f685d","#a06a5d","#a26c5d","#a36f5d","#a5715d","#a7735d","#a8765d","#aa785d","#ab7a5d","#ac7d5e","#ae7f5e","#af815f","#b1845f","#b28660","#b38861","#b58b62","#b68d64","#b78f65","#b89166","#b99468","#ba966a","#bb986c","#bc9b6e","#bd9d70","#be9f72","#bfa275","#c0a478","#c1a67b","#c2a87e","#c2ab81","#c3ad85","#c4af88","#c4b18c","#c5b490","#c5b695","#c6b899","#c6ba9e","#c7bca3","#c7bfa8","#c7c1ae","#c7c3b3","#c8c5b9","#c8c7c0","#c8c9c6","#c8cccd","#c8ced4","#c7d0db","#c7d2e3","#c7d4eb","#c7d6f3","#c6d8fb","#c6daff","#c5dcff","#c5deff","#c4e0ff","#c4e2ff","#c3e4ff","#c2e6ff","#c1e8ff","#c0eaff","#bfecff","#beeeff","#bdf0ff","#bcf2ff","#baf4ff","#b9f5ff","#b8f7ff","#b6f9ff","#b5fbff","#b3fdff","#b1feff"]')
lavhaze = LinearSegmentedColormap.from_list('lavhaze',_lavhaze)
allcms.append(lavhaze)

_candy = json.loads('["#00355e","#003560","#003461","#003363","#003365","#003267","#003269","#00316b","#00316d","#00306f","#003070","#003072","#003074","#002f76","#002f77","#002f79","#032f7b","#0c2f7c","#142f7e","#1c2f80","#242f81","#2c3083","#343084","#3b3086","#433087","#4a3189","#51318a","#58328c","#5f328d","#66338f","#6c3390","#723491","#793493","#7f3594","#853695","#8b3797","#903898","#963999","#9b399a","#a13a9c","#a63b9d","#ab3d9e","#af3e9f","#b43fa0","#b940a1","#bd41a2","#c143a3","#c544a4","#c945a5","#cd47a6","#d148a7","#d44aa8","#d84ba9","#db4daa","#de4fab","#e150ac","#e452ad","#e654ae","#e956ae","#eb58af","#ee59b0","#f05bb1","#f25db2","#f360b2","#f562b3","#f764b4","#f866b4","#f968b5","#fa6bb5","#fb6db6","#fc6fb7","#fd72b7","#fd74b8","#fd77b8","#fe79b9","#fe7cb9","#fe7eba","#fd81ba","#fd84ba","#fd87bb","#fc89bb","#fb8cbc","#fa8fbc","#f992bc","#f895bc","#f698bd","#f59bbd","#f39ebd","#f1a2bd","#f0a5be","#eda8be","#ebabbe","#e9afbe","#e6b2be","#e4b6be","#e1b9be","#debdbe","#dbc0be","#d8c4be","#d4c8be","#d1cbbe","#cdcfbe","#c9d3be","#c5d7be","#c1dbbe","#bddfbe","#b8e3be","#b4e7bd","#afebbd","#aaefbd","#a5f3bd","#a0f7bd","#9bfbbc","#96ffbc","#90ffbc","#8affbb","#85ffbb","#7fffba","#79ffba","#72ffba","#6cffb9","#65ffb9","#5fffb8","#58ffb8","#51ffb7","#4affb7","#42ffb6","#3bffb6"]')
candy = LinearSegmentedColormap.from_list('candy',_candy)
allcms.append(candy)
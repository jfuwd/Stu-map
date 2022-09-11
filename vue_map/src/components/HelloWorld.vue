
<template>

  <div class="hello">
    <transition name="bounce-in">
    <svg  width="4000px" height="4000px" version="1.1"
          xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
      <image href="../assets/map1.jpg" x="0" y="0" height="4000px" width="4000px"/>

      <line v-if="showline[0].show"   x1="3190" y1="3400" x2="2900" y2="3150"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[1].show" x1="3190" y1="3400" x2="2900" y2="3150"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[1].show" x1="2900" y1="3150" x2="2700" y2="3050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[2].show" x1="3190" y1="3400" x2="3300" y2="3250"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[3].show"   x1="2900" y1="3150" x2="2700" y2="3050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[4].show"  x1="3150" y1="3050" x2="3300" y2="3250"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[5].show" x1="3150" y1="3050" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[5].show" x1="3000" y1="2800" x2="3200" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[6].show"   x1="3150" y1="3050" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[6].show"  x1="3000" y1="2800" x2="2800" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[7].show" x1="3150" y1="3050" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[7].show" x1="3000" y1="2800" x2="3000" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[8].show"   x1="2700" y1="3050" x2="2350" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[8].show"  x1="2350" y1="2750" x2="2300" y2="2700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[9].show" x1="3300" y1="3250" x2="3180" y2="3050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[9].show" x1="2800" y1="2750" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[9].show"   x1="3190" y1="3400" x2="2900" y2="3150"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[10].show"  x1="3200" y1="2800" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[10].show" x1="3000" y1="2800" x2="3000" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[11].show" x1="2150" y1="3050" x2="2050" y2="2950"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[11].show"   x1="2050" y1="2950" x2="2350" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[11].show" x1="2350" y1="2750" x2="2300" y2="2700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[12].show"  x1="2150" y1="3050" x2="2050" y2="2950"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[12].show" x1="2050" y1="2950" x2="1800" y2="3100"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[12].show" x1="1800" y1="3100" x2="1775" y2="3000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[13].show"   x1="2800" y1="2750" x2="2650" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[13].show"  x1="2650" y1="2750" x2="2650" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[14].show" x1="2800" y1="2750" x2="3000" y2="2800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[14].show" x1="3000" y1="2800" x2="3000" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[15].show"   x1="2650" y1="2650" x2="2650" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[15].show"  x1="2650" y1="2750" x2="2350" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[15].show" x1="2350" y1="2750" x2="2300" y2="2700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[16].show" x1="3000" y1="2650" x2="2900" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[17].show"   x1="3000" y1="2650" x2="3120" y2="2650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[17].show"  x1="3120" y1="2650" x2="3120" y2="2500"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[18].show" x1="3120" y1="2500" x2="2950" y2="2400"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[19].show"  x1="2820" y1="2400" x2="2810" y2="2320"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[19].show" x1="2810" y1="2320" x2="2620" y2="2320"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[20].show"   x1="2620" y1="2320" x2="2620" y2="2500"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[20].show"  x1="2620" y1="2500" x2="2300" y2="2700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[21].show" x1="2620" y1="2320" x2="2620" y2="2270"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[21].show"   x1="2620" y1="2270" x2="2500" y2="2300"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[22].show"  x1="2620" y1="2320" x2="2550" y2="2020"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[22].show" x1="2550" y1="2020" x2="2400" y2="2000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[22].show" x1="2400" y1="2000" x2="2100" y2="2000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[23].show"   x1="2300" y1="2700" x2="2350" y2="2480"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[24].show"  x1="2300" y1="2700" x2="2050" y2="2520"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[25].show" x1="2350" y1="2480" x2="2500" y2="2300"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[26].show"   x1="2350" y1="2480" x2="2170" y2="2385"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[27].show"  x1="2350" y1="2480" x2="2050" y2="2520"  style="stroke:rgb(0,0,255);stroke-width:10"/>
    
      <line v-if="showline[28].show" x1="2500" y1="2300" x2="2350" y2="2480"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[28].show" x1="2350" y1="2480" x2="2170" y2="2385"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[29].show" x1="2500" y1="2300" x2="2350" y2="2050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[30].show" x1="2170" y1="2385" x2="2350" y2="2050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[31].show" x1="2050" y1="2520" x2="1750" y2="2520"  style="stroke:rgb(0,0,255);stroke-width:10"/>
     
      <line v-if="showline[32].show" x1="2050" y1="2520" x2="1820" y2="2220"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[32].show"   x1="1820" y1="2220" x2="1600" y2="2380"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[33].show"  x1="2050" y1="2520" x2="1820" y2="2220"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[33].show"   x1="1820" y1="2220" x2="1900" y2="2180"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[34].show" x1="1800" y1="2850" x2="1775" y2="3000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[35].show" x1="1800" y1="2850" x2="1600" y2="2670"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[36].show"  x1="1775" y1="3000" x2="1800" y2="3100"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[36].show"  x1="1800" y1="3100" x2="2050" y2="3100"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[37].show" x1="1775" y1="3000" x2="1600" y2="3120"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[38].show"  x1="1600" y1="3120" x2="1750" y2="2950"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[39].show"   x1="1600" y1="3120" x2="850" y2="3200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[39].show"  x1="850" y1="3200" x2="800" y2="3100"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[40].show" x1="1600" y1="2670" x2="1400" y2="2520"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[41].show" x1="800" y1="3100" x2="600" y2="3050"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[42].show"   x1="1400" y1="2520" x2="1395" y2="2580"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[43].show"  x1="1400" y1="2520" x2="1600" y2="2380"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[44].show" x1="1395" y1="2580" x2="1100" y2="2250"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[45].show" x1="1100" y1="2500" x2="900" y2="2450"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[46].show" x1="1100" y1="2500" x2="1100" y2="2250"  style="stroke:rgb(0,0,255);stroke-width:10"/>

      <line v-if="showline[47].show" x1="900" y1="2450" x2="975" y2="2230"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[48].show" x1="975" y1="2230" x2="1005" y2="2020"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[48].show" x1="1005" y1="2020" x2="940" y2="2060"  style="stroke:rgb(0,0,255);stroke-width:10"/>
     
      <line v-if="showline[49].show" x1="940" y1="2060" x2="950" y2="1975"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[49].show" x1="950" y1="1975" x2="1250" y2="1650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[49].show" x1="1250" y1="1650" x2="1350" y2="1750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[50].show" x1="940" y1="2060" x2="950" y2="1975"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[50].show" x1="950" y1="1975" x2="1250" y2="1650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[50].show" x1="1250" y1="1650" x2="1140" y2="1540"  style="stroke:rgb(0,0,255);stroke-width:10"/>
    
      <line v-if="showline[51].show" x1="1455" y1="2360" x2="1400" y2="2350"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[52].show" x1="1455" y1="2360" x2="1630" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>

      <line v-if="showline[53].show" x1="1455" y1="2360" x2="1820" y2="2220"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[53].show" x1="1820" y1="2220" x2="1900" y2="2180"  style="stroke:rgb(0,0,255);stroke-width:10"/>

      <line v-if="showline[54].show" x1="1400" y1="2350" x2="1350" y2="2300"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[55].show" x1="1400" y1="2350" x2="1630" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[56].show" x1="1350" y1="2300" x2="1470" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[57].show" x1="1350" y1="2300" x2="1350" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[58].show" x1="1630" y1="2200" x2="1470" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[59].show" x1="1470" y1="2200" x2="1350" y2="2200"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[60].show" x1="1200" y1="1880" x2="1350" y2="1750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[61].show" x1="1350" y1="1750" x2="1460" y2="1700"  style="stroke:rgb(0,0,255);stroke-width:10"/>

      <line v-if="showline[62].show" x1="1350" y1="1750" x2="1250" y2="1650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[62].show" x1="1250" y1="1650" x2="1310" y2="1500"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[62].show" x1="1310" y1="1500" x2="1540" y2="1480"  style="stroke:rgb(0,0,255);stroke-width:10"/>
    
      <line v-if="showline[63].show" x1="1350" y1="1750" x2="1250" y2="1650"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[63].show" x1="1250" y1="1650" x2="1140" y2="1540"  style="stroke:rgb(0,0,255);stroke-width:10"/> 
      
      <line v-if="showline[64].show" x1="1630" y1="2000" x2="1820" y2="2220"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[64].show" x1="1820" y1="2220" x2="1900" y2="2180"  style="stroke:rgb(0,0,255);stroke-width:10"/>
    
      <line v-if="showline[65].show" x1="1630" y1="2000" x2="1850" y2="1950"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[66].show" x1="1630" y1="2000" x2="1630" y2="1800"  style="stroke:rgb(0,0,255);stroke-width:10"/>    
      <line v-if="showline[67].show" x1="2100" y1="2000" x2="1950" y2="2000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[67].show" x1="1950" y1="2000" x2="1900" y2="1800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
    
      <line v-if="showline[68].show" x1="1850" y1="1950" x2="1900" y2="1800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[69].show" x1="1730" y1="1900" x2="1640" y2="1800"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[70].show" x1="1900" y1="1800" x2="1850" y2="1700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[71].show" x1="1630" y1="1800" x2="1580" y2="1750"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[72].show" x1="1630" y1="1800" x2="1460" y2="1700"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[73].show" x1="1580" y1="1750" x2="1700" y2="1680"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[74].show" x1="1460" y1="1700" x2="1550" y2="1600"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[75].show" x1="1850" y1="1700" x2="2050" y2="1700"  style="stroke:rgb(0,0,255);stroke-width:10"/> 
      <line v-if="showline[75].show" x1="2050" y1="1700" x2="1800" y2="1500"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[75].show" x1="1800" y1="1500" x2="1540" y2="1480"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[76].show" x1="1540" y1="1480" x2="1310" y2="1500"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[76].show" x1="1310" y1="1500" x2="1310" y2="1300"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[76].show" x1="1310" y1="1300" x2="1220" y2="1400"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[77].show" x1="1140" y1="1540" x2="1220" y2="1400"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[78].show" x1="1140" y1="1540" x2="940" y2="1540"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[79].show" x1="1220" y1="1400" x2="1310" y2="1300"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[79].show" x1="1310" y1="1300" x2="1170" y2="1160"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      
      <line v-if="showline[80].show" x1="1170" y1="1160" x2="1220" y2="1000"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[81].show" x1="1170" y1="1160" x2="1065" y2="1130"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[81].show" x1="1065" y1="1130" x2="1020" y2="900"  style="stroke:rgb(0,0,255);stroke-width:10"/>
      <line v-if="showline[82].show" x1="1220" y1="1000" x2="1240" y2="900"  style="stroke:rgb(0,0,255);stroke-width:10"/>

      <line v-if="showline[83].show" x1="3200" y1="2800" x2="2800" y2="2750"  style="stroke:rgb(0,0,255);stroke-width:10"/>

    </svg>
    </transition>

    <div class="box2">
      <h1 class="sub2">起点</h1>
    </div>
    <div class="box1 clearfix">
      <input v-model="data.begin" class="txt" >
    </div>

    <div class="box3">
      <h1 class="sub3">需要经过的点(例如，输入：2,3,5)</h1>
    </div>
    <div class="box1 clearfix">
      <input v-model="data.pass" class="txt">
    </div>

    <div class="box2">
      <h1 class="sub2">终点</h1>
    </div>
    <div class="box1 clearfix">
      <input v-model="data.endpoint" class="txt">
    </div>

    <div>
      <p><button v-on:click="select()" class="add_btn">确认</button></p>
    </div>

    <div class="box4" v-if="isshow">
      <h1 class="sub4">最短路径</h1>
    </div>
    <h1 class="txt">{{enddata.pass}}</h1>

    <div class="box4" v-if="isshow">
      <h1 class="sub4">实际通过景点</h1>
    </div>
    <h1 class="txt">{{enddata.truepass}}</h1>
<!--    <div><button v-on:click="change()" >更改</button></div>-->

  </div>



</template>

<script>






import axios from 'axios'
export default {
  data(){
    return{
      data:{
        begin:"",
        pass:"",
        endpoint:""
      },
      enddata:{
        pass:"",
        truepass:""
      },
      isshow:false,
      showline:[
        {id:"0102",show:false},
        {id:"0103",show:false},
        {id:"0104",show:false},
        {id:"0203",show:false},
        {id:"0204",show:false},
        {id:"0205",show:false},
        {id:"0207",show:false},
        {id:"0209",show:false},
        {id:"0314",show:false},
        {id:"0405",show:false},
        {id:"0509",show:false},
        {id:"0614",show:false},
        {id:"0622",show:false},
        {id:"0708",show:false},
        {id:"0709",show:false},
        {id:"0814",show:false},
        {id:"0910",show:false},
        {id:"0911",show:false},
        {id:"1112",show:false},
        {id:"1213",show:false},
        {id:"1314",show:false},
        {id:"1316",show:false},
        {id:"1346",show:false},
        {id:"1415",show:false},
        {id:"1419",show:false},
        {id:"1516",show:false},
        {id:"1517",show:false},
        {id:"1519",show:false},
        {id:"1617",show:false},
        {id:"1618",show:false},
        {id:"1718",show:false},
        {id:"1920",show:false},
        {id:"1936",show:false},
        {id:"1945",show:false},
        {id:"2122",show:false},
        {id:"2126",show:false},
        {id:"2223",show:false},
        {id:"2224",show:false},
        {id:"2425",show:false},
        {id:"2427",show:false},
        {id:"2629",show:false},
        {id:"2728",show:false},
        {id:"2930",show:false},
        {id:"2936",show:false},
        {id:"3033",show:false},
        {id:"3132",show:false},
        {id:"3133",show:false},
        {id:"3234",show:false},
        {id:"3435",show:false},
        {id:"3543",show:false},
        {id:"3557",show:false},
        {id:"3637",show:false},
        {id:"3639",show:false},
        {id:"3645",show:false},
        {id:"3738",show:false},
        {id:"3739",show:false},
        {id:"3840",show:false},
        {id:"3841",show:false},
        {id:"3940",show:false},
        {id:"4041",show:false},
        {id:"4243",show:false},
        {id:"4352",show:false},
        {id:"4356",show:false},
        {id:"4357",show:false},
        {id:"4445",show:false},
        {id:"4447",show:false},
        {id:"4450",show:false},
        {id:"4649",show:false},
        {id:"4749",show:false},
        {id:"4850",show:false},
        {id:"4954",show:false},
        {id:"5051",show:false},
        {id:"5052",show:false},
        {id:"5153",show:false},
        {id:"5255",show:false},
        {id:"5456",show:false},
        {id:"5658",show:false},
        {id:"5758",show:false},
        {id:"5759",show:false},
        {id:"5860",show:false},
        {id:"6061",show:false},
        {id:"6063",show:false},
        {id:"6162",show:false},
        {id:"0507",show:false},
      ],
    }

  },
  methods:{
    select(){
      for (var obj in this.showline){
          this.showline[obj]["show"]=false

        }
      console.log(this.data)//控制台查看
      axios({
        method: 'post',//post和get两种模式
        url:'http://localhost:5000/find' ,//使用本地前后端时使用
        //url: 'http://96.45.161.145:6001/find',
        data: this.data,
      }).then(resp =>{//接收数据
        // console.log(resp)
        console.log(resp.data.enddata)
        this.enddata=resp.data.enddata
        this.isshow=true
        this.show()
      })
    },
    show(){
      var showpass=this.enddata.truepass.split(",");
      console.log(showpass);
      for (let i = 0; i < showpass.length -1; i++) {
        // console.log(this.enddata.truepass[i]);
        let j=i+1;
        if(Number(showpass[i])>Number(showpass[j])){//如果前面的元素大于后面的元素
            var indexdata=new String();
            if(Number(showpass[i])<10 && Number(showpass[j])<10)
                indexdata="0"+showpass[j]+"0"+showpass[i]
            else if(Number(showpass[i])<10 &&Number(showpass[j])>=10)
                 indexdata=showpass[j]+"0"+showpass[i]
            else if(Number(showpass[j])<10 &&Number(showpass[i])>=10)
              indexdata="0"+showpass[j]+showpass[i]
            else
              indexdata=showpass[j]+this.showpass[i]
            console.log(indexdata)
            for (var obj in this.showline){
              if(this.showline[obj]["id"]==indexdata){


                  this.showline[obj]["show"]=true

              }
            }

        }else{
          var indexdata1=new String();
          if(Number(showpass[i])<10 && Number(showpass[j])<10)
            indexdata1="0"+showpass[i]+"0"+showpass[j]
          else if(Number(showpass[i])<10 &&Number(showpass[j])>=10)
            indexdata1="0"+showpass[i]+showpass[j]
          else if(Number(showpass[j])<10 &&Number(showpass[i])>=10)
            indexdata1=showpass[i]+"0"+showpass[j]
          else
            indexdata1=showpass[i]+showpass[j]
          console.log(indexdata1)
          for (var obj1 in this.showline){
            if(this.showline[obj1]["id"]==indexdata1){

                this.showline[obj1]["show"]=true

            }
          }
        }
      }
    },
    // change(){
    //   for (var obj in this.showline){
    //
    //             this.showline[obj]["show"]=true
    //
    //         }
    // }

  }



}

</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.line canvas{
  border:1px solid;
}
.fl{
  float: left;
}
.clearfix{
  clear: both;
  content: "";
  display: block;
}

.add_btn {
  min-width: 200px;
  min-height: 100px;
  font-size: 50px;

}
.box1{
  width: 629px;
  height: 80px;
  border: 2px solid black;
  border-radius: 16px;
  overflow: hidden;

  margin: auto;
  margin-top: 20px;
}
.box2{
  width: 200px;
  height: 80px;
  border: 2px solid black;
  border-radius: 16px;
  overflow: hidden;

  margin: auto;
  margin-top: 20px;

}
.box3{
  width: 800px;
  height: 80px;
  border: 2px solid black;
  border-radius: 16px;
  overflow: hidden;

  margin: auto;
  margin-top: 20px;
}
.box4{
  width: 400px;
  height: 80px;
  border: 2px solid black;
  border-radius: 16px;
  overflow: hidden;

  margin: auto;
  margin-top: 20px;
}
.box5{
  width: 400px;
  height: 100px;
  border: 2px solid black;
  border-radius: 16px;
  overflow: hidden;

  margin: auto;
  margin-top: 20px;
}
.txt{
  width: 532px;
  height: 80px;
  line-height: 70px;
  border: none;
  outline: none;
  padding-left: 10px;
  font-size: 60px;
  margin: auto;
}

.sub1{
  width: 75px;
  height: 36px;
  background-color: black;
  line-height: 36px;
  color: #ffffff;
  text-align: center;
  border: none;
  font-size: 16px;
  outline: none;
}
.sub2{
  width: 200px;
  height: 80px;

  line-height: 36px;
  color: black;
  text-align: center;
  border: none;
  font-size: 50px;
  outline: none;
}
.sub3{
  width: 800px;
  height: 80px;

  line-height: 36px;
  color: black;
  text-align: center;
  border: none;
  font-size: 50px;
  outline: none;
}
.sub4{
  width: 400px;
  height: 80px;

  line-height: 36px;
  color: black;
  text-align: center;
  border: none;
  font-size: 50px;
  outline: none;
}




h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}

</style>

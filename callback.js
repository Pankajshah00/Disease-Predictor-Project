function step1(callback){
    
}
function step2(callback){

}
function step3(callback){

}



step1(()=>(
    step2(()=>(
        step3()
    ))
    console.log("all done")
))
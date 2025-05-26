// let a = 5;
// let b=10;
// let res=a+b;
// async function getdata(){
//     let get = await fetch('https://jsonplaceholder.typicode.com/posts/1'); 
//     console.log(get);
// }
// getdata();
// console.log(res);

fetch('https://jsonplaceholder.typicode.com/posts/1')
.then((data)=>(
    console.log(data)
))
.catch((error)=>(
    console.log('error')
))
const express = require('express');
const app = express()

app.set("view engine", 'ejs')

app.get('/', (req,res)=>{
    res.send("This is my home page")
})
app.get('/login',(req,res)=>{
    res.render('login')
})

app.get('/signup',(req,res)=>{
    res.render('signup')
})

app.listen(3000)
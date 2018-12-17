var app = require('express')(),
	bodyParser = require('body-parser'),
    server = require('http').createServer(app),
    io = require('socket.io').listen(server),
    ent = require('ent'), // Permet de bloquer les caractères HTML (sécurité équivalente à htmlentities en PHP)
    fs = require('fs');
const Window = require('window');
var tf = require('@tensorflow/tfjs');
var cors = require('cors');
var express = require('express');
//var frozen = require('@tensorflow/tfjs-converter');


app.use(cors())
app.use(express.static('/js'))
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({     // to support URL-encoded bodies
  extended: true
})); 




io.sockets.on('connection', function (socket, pseudo) {
    // Dès qu'on nous donne un pseudo, on le stocke en variable de session et on informe les autres personnes
    socket.on('object', function(name,obj) {
        console.log(obj);
        fs.writeFile(name+'.json', obj, (err) => {
  			if (err) throw err;
  			console.log('The file has been saved!');
		});
    });
});

app.get('/', function (req, res) {
    fs.readFile("index.html", function (error, pgResp) {
        if (error) {
            res.writeHead(404);
            res.write('Contents you are looking are Not Found');
        } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.write(pgResp);   
        }    
        res.end();
    });
});

app.get('/model.js', function (req, res) {
    //Window.onload=func;
    fs.readFile("model.js", function (error, pgResp) {
        if (error) {
            res.writeHead(404);
            res.write('Contents you are looking are Not Found');
        } else {
            res.writeHead(200, { 'Content-Type': 'text/json' });
            res.write(pgResp);   

        }    
        res.end();
        
    });
});

app.get('/haarcascade_frontalface_default.xml', function (req, res) {
    //Window.onload=func;
    fs.readFile("haarcascade_frontalface_default.xml", function (error, pgResp) {
        if (error) {
            res.writeHead(404);
            res.write('Contents you are looking are Not Found');
        } else {
            res.writeHead(200, { 'Content-Type': 'text/xml' });
            res.write(pgResp);   

        }    
        res.end();
        
    });
});

app.get('/js/weights_manifest.json', function (req, res) {
    //Window.onload=func;
    fs.readFile("js/weights_manifest.json", function (error, pgResp) {
        if (error) {
            res.writeHead(404);
            res.write('Contents you are looking are Not Found');
        } else {
            res.writeHead(200, { 'Content-Type': 'text/json' });
            res.write(pgResp);   
            //var file = "/js/weights_manifest.json"
            //res.send(pgResp)
        }  
        res.end();
        
    });
});

app.get('/js/tensorflowjs_model.pb', function (req, res) {
    //Window.onload=func;
    fs.readFile("js/tensorflowjs_model.pb", function (error, pgResp) {
        if (error) {
            res.writeHead(404);
            res.write('Contents you are looking are Not Found');
        } else {
            res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
            res.write(pgResp);   
            //console.log("blabla 2")
            //var file = "/js/tensorflowjs_model.pb"
            //res.send(pgResp)
        }  
        res.end();
        
    });
    
});

for (let i = 1; i <= 22; i++) {
    app.get('/js/group1-shard'+i+'of22', function (req, res) {
        //Window.onload=func;
        fs.readFile("js/group1-shard"+i+"of22", function (error, pgResp) {
            console.log("js/group1-shard"+i+"of22")
            if (error) {
                res.writeHead(404);
                res.write('Contents you are looking are Not Found');
            } else {
                res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
                res.write(pgResp);   
                //console.log("blabla 2")
                //var file = "/js/tensorflowjs_model.pb"
                //res.send(pgResp)
            }   
            res.end();
            
        });
    
    });
};


for (let j = 1; j <= 9; j++) {
    app.get('/js/group1-shard'+j+'of9', function (req, res) {
        //Window.onload=func;
        fs.readFile("js/group1-shard"+j+"of9", function (error, pgResp) {
            console.log("js/group1-shard"+j+"of9")
            if (error) {
                res.writeHead(404);
                res.write('Contents you are looking are Not Found');
            } else {
                res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
                res.write(pgResp);   
                //console.log("blabla 2")
                //var file = "/js/tensorflowjs_model.pb"
                //res.send(pgResp)
            }   
            res.end();
            
        });
    
    });
};

server.listen(8080);
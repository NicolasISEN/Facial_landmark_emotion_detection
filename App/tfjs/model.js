

var init_model = async function(){
	//console.log("Test");
	//const model = await tf.loadModel('model.json');
	/*var a = $.get( "js/tensorflowjs_model.pb", function( data ) {
  		alert( "Load was performed." );
	});
	var b = $.getJSON("js/weights_manifest.json", function(data) {
		//console.log(data.url);
	});
	console.log(a)*/
	const MODEL_URL = 'http://localhost:8080/js/tensorflowjs_model.pb';
	const WEIGHTS_URL = 'http://localhost:8080/js/weights_manifest.json';
	console.log('blabla')
	console.log(tf.version)
	const model =  await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL,{credentials: 'include'});
	console.log("1")
	//const cat = document.getElementById('cat');
	//model.execute({input: tf.fromPixels(cat)});
	console.log(model)
	//model.execute({input: tf.tensor2d(Array.from({length: 128*128}, () => Math.floor(Math.random()*255)), [128, 128])});
	var pred = model.predict(tf.tensor4d(Array.from({length: 128*128}, () => Math.floor(Math.random()*255)), [1,1,128, 128]));
	console.log(pred[0].print())
	console.log(pred[1].print())

}
init_model();
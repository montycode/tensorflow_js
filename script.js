let model;

const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const imgEl = document.getElementById("img");
const descEl = document.getElementById("descripcion_imagen");

var count = 0;
var net ;
var webcam;

async function app() {
    //App
	console.log("Cargando modelo de identificacion de imagenes");
    net = await mobilenet.load(); // Cargamos con mobilenet un modelo de identificacion de imagenes (Esperamos con await).
	console.log("Carga terminada.")
    
    // Clasificamos la imagen de carga
	const result = await net.classify(imgEl);
	console.log(result);
    descEl.innerHTML= JSON.stringify(result);

    // Obtenemos datos del webcam
    webcam = await tf.data.webcam(webcamElement);
    
    // Y los vamos procesando por medio de un ciclo
    while (true) {
        const img = await webcam.capture(); // Accede al webcam
        const result = await net.classify(img); 
        const activation = net.infer(img, 'conv_preds');

        var result2;

        try {
            result2 = await classifier.predictClass(activation);
        } 
        catch (error) {
            result2 = {};
        }

        const classes = ["Sin Entrenar", "Control", "Reloj" , "Omar", "OK","Rock"]

        document.getElementById('console').innerText = `
        Predicción: ${result[0].className}\n
        Probabilidad: ${result[0].probability}
        `;

        try {
            document.getElementById("console2").innerText = `
            Predicción: ${classes[result2.label]}\n
            Probabilidad: ${result2.confidences[result2.label]}
            `;
        } catch (error) {
            document.getElementById("console2").innerText="Sin Entrenar";
        }

        // Desecha el tensor para liberar la memoria.
        img.dispose();

        // Dé un poco de espacio para respirar esperando a que se dispare el siguiente cuadro de animación
        await tf.nextFrame();
    }
}

img.onload = async function() {
   try {
     result = await net.classify(img);
     descEl.innerHTML= JSON.stringify(result);
   } catch (error) {
   }
}

async function cambiarImagen() {
    count = count + 1;
    imgEl.src = "https://picsum.photos/200/300?random=" + count;
    descEl.innerHTM = "";
}

// Añadir ejemplo
async function addExample (classId) {
  const img = await webcam.capture();
  const activation = net.infer(img, true);
  classifier.addExample(activation, classId);
  img.dispose() // Liberamos el tensor
}

const saveKnn = async () => {
    // Obtenemos el dataset actual del clasificador (labels y vectores)
    let strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]));
    const storageKey = "knnClassifier";
    // Lo almacenamos en el localStorage
    localStorage.setItem(storageKey, strClassifier);
};

const loadKnn = async ()=>{
    const storageKey = "knnClassifier";
    let datasetJson = localStorage.getItem(storageKey);
    classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])));
};


app()

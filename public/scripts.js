console.log(faceapi)

const run = async()=>{
    //we need to load our models

    //loading the models is going to use await
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    })
    const videoFeedE1 = document.getElementById('video-feed')
    videoFeedE1.srcObject = stream

    // pre-trained models
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models')
    ])

    //make canvas same size as video feed
    const canvas = document.getElementById('canvas')
    canvas.style.left = videoFeedE1.offsetLeft
    canvas.style.top = videoFeedE1.offsetTop
    canvas.height = videoFeedE1.height
    canvas.width = videoFeedE1.width

    const refFace = await faceapi.fetchImage('https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg/220px-Cristiano_Ronaldo_playing_for_Al_Nassr_FC_against_Persepolis%2C_September_2023_%28cropped%29.jpg')

    let refFaceAIDATA = await faceapi.detectAllFaces(refFace).withFaceLandmarks().withFaceDescriptors()

    let faceMatcher = new faceapi.FaceMatcher(refFaceAIDATA)

    // face detection with points
    setInterval(async()=>{
        let faceAIData = await faceapi.detectAllFaces(videoFeedE1).withFaceLandmarks().withFaceDescriptors().withAgeAndGender().withFaceExpressions()

        //clear the canvas
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height)


        faceAIData = faceapi.resizeResults(faceAIData, videoFeedE1)
        faceapi.draw.drawDetections(canvas, faceAIData)
        faceapi.draw.drawFaceLandmarks(canvas, faceAIData)
        faceapi.draw.drawFaceExpressions(canvas, faceAIData)

        faceAIData.forEach(face => {
            const {age, gender, genderProbability, detection, descriptor} = face
            const genderText = `${gender} - ${Math.round(genderProbability * 100)/100*100}`
            const ageText = `${Math.round(age)} years old`
            const textField = new faceapi.draw.DrawTextField(
                [genderText, ageText],
                face.detection.box.bottomRight,
                {
                    color: 'white',
                    fontSize: 16
                }
            )
            textField.draw(canvas)

            let label = faceMatcher.findBestMatch(descriptor).toString()
            console.log(label)

            let options = {label: "Ronaldo"}
            if(label.includes("unknown")){
                options = {label: "Unknown"}
            }
            else if(label.includes("Cristiano")){
                options = {label: "Cristiano"}
            }

            const drawBox = new faceapi.draw.DrawBox(detection.box, options)
            drawBox.draw(canvas)
        })
    }, 500)

}

run()
<<<<<<< HEAD
let timer = 0
let interval

function startLoading() {

    document.getElementById("loader").style.display = "block"

    timer = 0

    interval = setInterval(() => {

        timer++
        document.getElementById("timer").innerText = timer

    }, 1000)

=======
let timer = 0
let interval

function startLoading() {

    document.getElementById("loader").style.display = "block"

    timer = 0

    interval = setInterval(() => {

        timer++
        document.getElementById("timer").innerText = timer

    }, 1000)

>>>>>>> 2c9ce2edacf97c226fcd3b78d981c9722c548384
}
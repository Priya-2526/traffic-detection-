let timer = 0
let interval

function startLoading() {

    document.getElementById("loader").style.display = "block"

    timer = 0

    interval = setInterval(() => {

        timer++
        document.getElementById("timer").innerText = timer

    }, 1000)

}
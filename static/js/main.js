/* global $ */
$(() => {

    $('#analyze').click(() => {
        var passage = $('#passage_text').val();
        var question = $('#question_text').val();
        console.log("passage:"+passage);
        console.log("question:"+question);
        var inputs = {
                "passage": passage ,
                "question": question
            };
        console.log(JSON.stringify(inputs));
        $.ajax({
            url: '/api/intqa',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(inputs),
            success: (data) => {
                console.log(data.results)
                $('#answer').text(data.results);
            }
        });
    });

});

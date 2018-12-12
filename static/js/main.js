/* global $ */
$(() => {

    $('#analyze').click(() => {
        var inputs = {
                passage: ""
                , question: ""
            };
        inputs["passage"] = $('#passage_text').text;
        inputs["question"] = $('#question_text').text;
        alert(JSON.stringify(inputs));
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

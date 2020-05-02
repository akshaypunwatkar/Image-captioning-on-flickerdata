$(document).ready(function () {
    // Init
<<<<<<< HEAD
=======
    console.log("I am ready")
>>>>>>> master
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
<<<<<<< HEAD
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
=======
              console.log(e)
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').show();
>>>>>>> master
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
<<<<<<< HEAD
    $("#imageUpload").change(function () {
=======
    $("#file").change(function () {
        console.log('Yum here')
>>>>>>> master
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(' Caption:  ' + data);
                console.log('Success!');
            },
        });
    });

<<<<<<< HEAD
});
=======
});
>>>>>>> master

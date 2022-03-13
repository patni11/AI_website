alert("HIIIII")
function detect_faces(){

	document.getElementById("detect").value = "Han on";
	var python = require("python-shell");
	var path = require("path");
	var options = {
		scriptPath: path.join(_dirname,'obj')

	}
	print("I am working")
	var face = new python("obj.py",options);
	face.end(function(err,code,message){
		document.getElementById("detect").value = "detect_onjs";
	})
	return "I am goof"
}

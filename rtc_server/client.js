// peer connection
pc_original = null;
pc_pose = null;
pc_appearance = null;
pcs = [pc_original, pc_pose, pc_appearance];


function createPeerConnection(transform) {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    config.iceServers = [{urls: ['stun:stun.l.google.com:19302']}];

    var pc = new RTCPeerConnection(config);

    // connect video
    pc.addEventListener('track', function(evt) {
        if (evt.track.kind == 'video')
            target = transform + "-video";
            document.getElementById(target).srcObject = evt.streams[0];}
    );

    return pc;
}


function negotiate(pc, transform) {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        var codec;

        codec = "default";

        offer_name = '/offer_' + transform

        // What we send back to the server
        return fetch(offer_name, {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                transform: transform,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}


function start() {
    pc_original = createPeerConnection('original');
    pc_pose = createPeerConnection('pose');
    pc_appearance = createPeerConnection('appearance');

    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }

    var constraints = {
        video: true
    };

    document.getElementById('media1').style.display = 'block';
    document.getElementById('media2').style.display = 'block';
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        stream.getTracks().forEach(function(track) {
            pc_original.addTrack(track, stream);
            pc_pose.addTrack(track, stream);
            pc_appearance.addTrack(track, stream);
        });
        negotiate(pc_original, 'original');
        negotiate(pc_pose, 'pose');
        negotiate(pc_appearance, 'appearance');
    }, function(err) {
        alert('Could not acquire media: ' + err);
    });

    document.getElementById('stop').style.display = 'inline-block';
}


function switch_app_img(img_name) {
	var image_path = "/assets/" + img_name
	return fetch('/switch_app_img', {
		body: JSON.stringify({
            img_name: img_name
		}),
		headers: {
			'Content-Type': 'application/json'
		},
		method: 'POST'
	});
}


function stop() {
    document.getElementById('stop').style.display = 'none';

    // close transceivers for all pcs
    pcs.forEach(pc => {
        if (pc.getTransceivers) {
            pc.getTransceivers().forEach(function(transceiver) {
                if (transceiver.stop) {
                    transceiver.stop();
                }
            });
        }

        // close local video
        pc.getSenders().forEach(function(sender) {
            sender.track.stop();
        });

        // close peer connection
        setTimeout(function() {
            pc.close();
        }, 500);})
}

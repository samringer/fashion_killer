// peer connection
var pc_original = null;
var pc_pose = null;
pcs = [pc_original, pc_pose];


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
        video: false
    };

    var resolution = document.getElementById('video-resolution').value;
    if (resolution) {
        resolution = resolution.split('x');
        constraints.video = {
            width: parseInt(resolution[0], 0),
            height: parseInt(resolution[1], 0)
        };
    } else {
        constraints.video = true;
    }

    if (constraints.video) {
        if (constraints.video) {
            document.getElementById('media').style.display = 'block';
        }
        navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
            stream.getTracks().forEach(function(track) {
                pc_original.addTrack(track, stream);
                pc_pose.addTrack(track, stream);
            });
            negotiate(pc_original, 'original');
            return negotiate(pc_pose, 'pose');
        }, function(err) {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate(pc_original, 'original');
        negotiate(pc_pose, 'pose');
    }

    document.getElementById('stop').style.display = 'inline-block';
}

document.addEventListener('DOMContentLoaded', function(){
                                            start()
                                            }, false);


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

using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

[Serializable]
public class Landmark {
    public int id;
    public float x, y, z;
    public float visibility;
}

[Serializable]
public class PoseMessage {
    public double timestamp;
    public List<Landmark> pose_landmarks;
    public List<Landmark> pose_world_landmarks;
}

public class PoseReceiver : MonoBehaviour {
    public string serverUrl = "ws://localhost:8765";
    private WebSocket websocket;

    public Transform[] jointTransforms; // assign 33 transforms in Inspector

    async void Start() {
        Debug.Log("Unity client starts!");
        try
        {
            websocket = new WebSocket(serverUrl);

            websocket.OnOpen += () => {
                Debug.Log($"WebSocket connected {serverUrl}");
            };
            websocket.OnError += (e) => {
                Debug.LogError("WebSocket Error: " + e);
            };
            websocket.OnClose += (e) => {
                Debug.Log("WebSocket closed: " + e);
            };
            websocket.OnMessage += (bytes) => {
                string msg = System.Text.Encoding.UTF8.GetString(bytes);
                HandleMessage(msg);
            };

            Debug.Log("Connecting to " + serverUrl + " ...");
            await websocket.Connect();
        }
        catch (Exception ex)
        {
            Debug.LogError("Failed to start WebSocket: " + ex.Message);
        }
    }

    private void HandleMessage(string msg)
    {
        try {
            PoseMessage poseMsg = JsonUtility.FromJson<PoseMessage>(msg);

            if (poseMsg.pose_world_landmarks != null && poseMsg.pose_world_landmarks.Count > 0)
            {
                var first = poseMsg.pose_world_landmarks[0];
                Debug.Log($"[PoseReceiver] Received pose_world_landmarks. First landmark: id={first.id} x={first.x:F3} y={first.y:F3} z={first.z:F3}");

                UpdateJoints(poseMsg.pose_world_landmarks);
            }
            else
            {
                Debug.Log("[PoseReceiver] No pose_world_landmarks found in message");
            }
        }
        catch (Exception ex) {
            Debug.LogWarning("JSON parse error: " + ex.Message);
        }
    }
    private void UpdateJoints(List<Landmark> landmarks) {
        if (jointTransforms == null || jointTransforms.Length == 0) return;

        foreach (var lm in landmarks) {
            int idx = lm.id;
            if (idx < jointTransforms.Length && jointTransforms[idx] != null) {
                Vector3 pos = new Vector3(lm.x, lm.y, -lm.z);
                jointTransforms[idx].localPosition = pos;
            }
        }
    }

    private async void OnApplicationQuit() {
        if (websocket != null) {
            await websocket.Close();
        }
    }

    void Update() {
        #if !UNITY_WEBGL || UNITY_EDITOR
            websocket?.DispatchMessageQueue();
        #endif
    }
}

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
    public List<Landmark> pose_world_landmarks;
}

public class PoseReceiver : MonoBehaviour {
    public string serverUrl = "ws://192.168.1.100:8765"; // your PC’s IP & port
    private WebSocket websocket;

    public Transform[] jointTransforms; // assign 33 transforms in Inspector

    async void Start() {
        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () => {
            Debug.Log("WebSocket connected");
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

        await websocket.Connect();
    }

    private void HandleMessage(string msg) {
        try {
            PoseMessage pm = JsonUtility.FromJson<PoseMessage>(msg);
            if (pm.pose_world_landmarks != null) {
                UpdateJoints(pm.pose_world_landmarks);
            }
        } catch (Exception ex) {
            Debug.LogWarning("JSON parse error: " + ex.Message);
        }
    }

    private void UpdateJoints(List<Landmark> landmarks) {
        foreach (var lm in landmarks) {
            int idx = lm.id;
            if (idx < jointTransforms.Length && jointTransforms[idx] != null) {
                // convert from meter space / normalized space to Unity units
                // Example simple mapping:
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
            websocket.DispatchMessageQueue();
        #endif
    }
}

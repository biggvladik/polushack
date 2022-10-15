import { FC, useEffect, useState, useRef } from "react";

export const Home: FC = () => {
  const streamImageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const socket = new WebSocket("ws://localhost:80/ws");

    socket.onmessage = (event) => {
      const image = event.data;

      if (streamImageRef.current) {
        streamImageRef.current.src = `data:image/png;base64,${image}`;
      }
    };
  }, []);

  return (
    <div className="page">
      <h1>Взлом золота</h1>
      <img ref={streamImageRef} src="" alt="" className="stream" />
    </div>
  );
};

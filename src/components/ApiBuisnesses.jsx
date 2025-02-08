import { useState } from "react";
import Navbar from "./Navbar/navbar2";

export default function API_Buisnesses() {
    const [LoaderActive, setLoaderActive] = useState(false);

    return (
        <div className="relative flex flex-col justify-center items-center h-screen">
            <Navbar active={3}/>
            {LoaderActive && (
                <div className="absolute inset-0 z-50 flex justify-center items-center bg-black bg-opacity-60">
                    <Loader size={64} className="animate-spin text-white" />
                    <div className="text-white text-2xl font-semibold ml-4">Loading ...</div>
                </div>
            )}
            {/* Aligning heading with navbar */}
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">API for Businesses</h1>
                <p className="text-2xl font-light">Empower your platform with Deepfake Detection and Blockchain Verification</p>
            </div>
            <div className={`w-[85%] px-12 mt-20 ${LoaderActive ? "opacity-50" : ""}`}>
                <h1 className="text-3xl font-semibold">Why choose DeepTrace API ?</h1>
                <div className="flex gap-10 pt-6">
                    <div className="p-4 rounded-2xl bg-[#252525]">
                        <div>
                            <div className="text-2xl font-semibold">
                            State-of-the-Art AI <br /> Detection
                            </div>
                            <div className="pt-2 text-xl font-light">
                                Detect AI-generated <br/> deepfakes with high accuracy <br/> using multi-modal analysis.
                            </div>
                        </div>
                    </div>
                    <div className="p-4 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Blockchain-powered <br /> Verification
                            </div>
                            <div className="pt-2 text-xl font-light">
                                Securely store and verify <br/> content authenticity using <br/> Hyperledger technology.
                            </div>
                        </div>
                    </div>
                    <div className="p-4 rounded-2xl bg-[#252525]">
                    <div>
                            <div className="text-2xl font-semibold">
                            Customizable & <br /> Adaptive
                            </div>
                            <div className="pt-2 text-xl font-light">
                                Custom APIs specially curated <br/> as per the custom buisness <br/> databases.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

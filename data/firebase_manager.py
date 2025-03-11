"""Firebase manager for handling Firebase Storage operations."""

import os
import logging
from typing import Set, Tuple
import firebase_admin
from firebase_admin import credentials, storage
from config.config_settings import FIREBASE_BUCKET, FIREBASE_PREFIX, DOWNLOAD_DIR

logger = logging.getLogger(__name__)

class FirebaseManager:
    """Manager for Firebase Storage operations."""
    
    def __init__(self):
        """Initialize Firebase manager with credentials from environment."""
        self.bucket = None
        self.initialized = False
    
    def initialize(self):
        """Initialize Firebase application if not already initialized."""
        if self.initialized:
            return
        
        try:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not cred_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            
            # Check if Firebase app is already initialized
            try:
                app = firebase_admin.get_app()
            except ValueError:
                # Initialize new app if not exists
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': FIREBASE_BUCKET
                })
            
            self.bucket = storage.bucket()
            self.initialized = True
            logger.info("Firebase initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    def get_existing_files(self) -> Set[str]:
        """
        Get set of existing local file paths relative to download directory.
        
        Returns:
            Set[str]: Set of relative file paths
        """
        existing_files = set()
        download_path = DOWNLOAD_DIR
        
        for root, _, files in os.walk(download_path):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), download_path)
                # Ensure cross-platform compatibility
                existing_files.add(relative_path.replace("\\", "/"))
        
        return existing_files
    
    def get_firebase_files(self) -> Set[str]:
        """
        Get set of all files in Firebase Storage with the specified prefix.
        
        Returns:
            Set[str]: Set of file paths in Firebase
        """
        self.initialize()
        blobs = [blob.name.replace(FIREBASE_PREFIX, "", 1) 
                for blob in self.bucket.list_blobs() 
                if blob.name.startswith(FIREBASE_PREFIX)]
        return set(blobs)
    
    def download_new_files(self) -> Tuple[int, Set[str]]:
        """
        Download only new files from Firebase.
        
        Returns:
            Tuple[int, Set[str]]: Number of new files downloaded and set of new file paths
        """
        self.initialize()
        existing_files = self.get_existing_files()
        
        # List and filter only files with the specified prefix
        blobs = [blob for blob in self.bucket.list_blobs() 
                if blob.name.startswith(FIREBASE_PREFIX)]
        
        new_files_downloaded = 0
        new_files = set()
        
        # Download only new files
        for blob in blobs:
            relative_blob_path = blob.name.replace(FIREBASE_PREFIX, "", 1)
            local_file_path = os.path.join(DOWNLOAD_DIR, relative_blob_path)
            
            # Check if file already exists
            if relative_blob_path in existing_files:
                logger.debug(f"Skipping (Already Exists): {blob.name}")
                continue
            
            # Ensure the folder structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the file
            blob.download_to_filename(local_file_path)
            logger.info(f"Downloaded: {blob.name} → {local_file_path}")
            new_files_downloaded += 1
            new_files.add(relative_blob_path)
        
        if new_files_downloaded == 0:
            logger.info("No new files found. Dataset is already up-to-date!")
        else:
            logger.info(f"{new_files_downloaded} new files downloaded successfully!")
        
        return new_files_downloaded, new_files

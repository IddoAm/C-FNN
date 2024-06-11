#include <stdio.h>
#include <windows.h> // For Sleep() function

int main() {
    const char* animation[] = {"|", "/", "-", "\\"};
    const int animation_frames = sizeof(animation) / sizeof(animation[0]);

    printf("Processing... ");

    // Infinite loop for continuous animation
    while (1) {
        for (int i = 0; i < animation_frames; i++) {
            printf("\b%s", animation[i]); // Move the cursor back and print the next animation frame
            fflush(stdout); // Flush the output buffer to ensure the message is printed immediately
            Sleep(500); // Sleep for 0.5 seconds (500 milliseconds)
        }
    }

    return 0;
}
